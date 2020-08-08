import argparse
import os
import json
import time
import random
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

  # Sample one
  # pid = random.choice(player_ids)
  # player_ids = [pid]

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
  with Pool() as pool:
    for grads_list in pool.imap_unordered(compute_grad, gen_args(replay_jsons)):
      all_grads_list.extend(grads_list)

  def apply_grad(grads_list):
    import train
    trainer = train.Trainer(None, model_dir)
    for i, grads in enumerate(grads_list):
      trainer.apply_grad(grads)
      print("apply grad at %s" % i)
    trainer.on_batch_finished()

  random.shuffle(all_grads_list)
  apply_grad(all_grads_list)


def replay_to_ship_rewards(replay_json):
  import train
  import numpy as np

  num_players = len(replay_json['rewards'])

  stats = np.zeros(2) # total deposit, total collect
  total_deposits = []
  total_collects = []
  returns = []
  for player_id in range(num_players):
    boards = train.gen_player_states(replay_json, player_id)
    boards = list(boards)

    ship_rewards = train.compute_returns(list(boards), as_list=True)
    returns = np.concatenate(list(ship_rewards.values()) + [returns])

    b = boards[-1]
    total_deposits.append(b.total_deposite)
    total_collects.append(b.total_collect)
  return returns, total_deposits, total_collects


def get_normalization_params(replays):
  import numpy as np

  returns = []
  total_deposits = []
  total_collects = []
  with Pool() as pool:
    for r, d, c in pool.imap_unordered(replay_to_ship_rewards, replays):
      returns = np.concatenate([returns, r])
      total_deposits = np.concatenate([total_deposits, d])
      total_collects = np.concatenate([total_collects, c])

  mean_return = np.mean(returns)
  std_return = np.std(returns)
  print("****Batch returns finished: ship reward(mean=%.2f, std=%.2f)"
        % (mean_return, std_return))

  D = np.mean(total_deposits)
  C = np.mean(total_collects)
  print("****Avg deposite = %.3f, avg collect = %.3f, ratio=%.5f"
        % (D, C, (D / (C  +1.0))))
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

