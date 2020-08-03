import argparse
import os
import json
import time
from multiprocessing import Pool

from lsd import BATCH_SIZE


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def scan_for_replays(episode_dir):
  while True:
    finished_count = 0
    if not os.path.exists(episode_dir):
      print("Waiting for", episode_dir)
      time.sleep(1)
      continue

    for name in os.listdir(episode_dir):
      if not name.endswith('.json'):
        continue
      if name.startswith('_'):
        finished_count += 1
        continue

      episode_path = os.path.join(episode_dir, name)
      print("Loading:", episode_path)
      with open(episode_path, 'r') as f:
        replay_json = json.loads(f.read())
      yield replay_json

      dest_path = os.path.join(episode_dir, '_' + name)
      os.rename(episode_path, dest_path)
      finished_count += 1

    if finished_count >= BATCH_SIZE:
      break
    time.sleep(1)
    print("Wait for more episodes...")


def compute_grad(args):
  import train

  replay_json, player_id, model_dir = args
  boards = list(train.gen_player_states(replay_json, player_id))
  trainer = train.Trainer(None, model_dir)
  return trainer.train([boards], apply_grad=False)


def train_on_replays_multiprocessing(model_dir, replay_jsons):

  def gen_args(replay_jsons):
    for replay_json in replay_jsons:
      num_players = len(replay_json['rewards'])
      player_ids = list(range(num_players))
      for player_id in player_ids:
        yield replay_json, player_id, model_dir


  all_grads_list = []
  with Pool(processes=5) as pool:
    for grads_list in pool.imap_unordered(compute_grad, gen_args(replay_jsons)):
      all_grads_list.extend(grads_list)

  def apply_grad(grads_list):
    import train
    trainer = train.Trainer(None, model_dir)
    for grads in grads_list:
      trainer.apply_grad(grads)
    trainer.on_batch_finished()

  apply_grad(all_grads_list)


def run_train(episode_dir, model_dir):
  replays = scan_for_replays(episode_dir)
  train_on_replays_multiprocessing(model_dir, replays)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  args = parser.parse_args()
  run_train(args.episode_dir, args.model_dir)


if __name__ == "__main__":
  main()

