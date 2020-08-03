import argparse
import os
import json
import time
from multiprocessing import Pool

from lsd import BATCH_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def scan_for_replays(episode_dir):
  finished_count = 0
  for name in os.listdir(episode_dir):
    episode_path = os.path.join(episode_dir, name)
    print("Loading:", episode_path)
    with open(episode_path, 'r') as f:
      replay_json = json.loads(f.read())
    yield replay_json
    finished_count += 1

  print("Total of %s episodes found." % finished_count)


def compute_grad(args):
  replay_json, model_dir, return_params = args

  import train

  num_players = len(replay_json['rewards'])
  player_ids = list(range(num_players))

  player_boards = [
    list(train.gen_player_states(replay_json, player_id))
    for player_id in player_ids
  ]
  trainer = train.Trainer(None, model_dir, return_params=return_params)
  return trainer.train(player_boards, apply_grad=False)


def train_on_replays_multiprocessing(model_dir, replay_jsons, return_params):
  def gen_args(replay_jsons):
    for replay_json in replay_jsons:
      yield replay_json, model_dir, return_params

  all_grads_list = []
  with Pool(processes=6) as pool:
    for grads_list in pool.imap_unordered(compute_grad, gen_args(replay_jsons)):
      all_grads_list.extend(grads_list)

  def apply_grad(grads_list):
    import train
    trainer = train.Trainer(None, model_dir)
    for i, grads in enumerate(grads_list):
      trainer.apply_grad(grads)
      print("apply grad at %s" % i)
    trainer.on_batch_finished()

  apply_grad(all_grads_list)


def compute_returns(replay_json):
  import train
  import numpy as np

  num_players = len(replay_json['rewards'])

  returns = []
  for player_id in range(num_players):
    boards = train.gen_player_states(replay_json, player_id)
    returns = np.concatenate([returns, train.compute_returns(boards)])
  return returns


def get_normalization_params(replays):
  import numpy as np

  returns = []
  with Pool(processes=6) as pool:
    for r in pool.imap_unordered(compute_returns, replays):
      returns = np.concatenate([returns, r])

  mean_return = np.mean(returns)
  std_return = np.std(returns)
  print("****Batch returns finished: mean=%.1f, std=%.1f"
        % (mean_return, std_return))
  return mean_return, std_return


def run_train(episode_dir, model_dir):
  replays = list(scan_for_replays(episode_dir))
  return_params = get_normalization_params(replays)
  train_on_replays_multiprocessing(model_dir, replays, return_params)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  args = parser.parse_args()
  run_train(args.episode_dir, args.model_dir)


if __name__ == "__main__":
  main()

