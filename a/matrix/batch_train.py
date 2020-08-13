import argparse
import os
import json
import time
import random
import numpy as np
from multiprocessing import Pool

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
  replay_json, model_dir = args

  import train

  num_players = len(replay_json['rewards'])
  player_ids = list(range(num_players))

  total_deposits = []
  total_collects = []
  grads = []
  trainer = train.Trainer(None, model_dir)
  for player_id in player_ids:
    boards = list(train.gen_player_states(replay_json, player_id))
    g = trainer.train([boards], apply_grad=False)
    grads.extend(g)

    b = boards[-1]
    total_deposits.append(b.total_deposite)
    total_collects.append(b.total_collect)
  return grads, total_deposits, total_collects


def train_on_replays_multiprocessing(model_dir, replay_jsons):
  def gen_args(replay_jsons):
    for replay_json in replay_jsons:
      yield replay_json, model_dir

  total_deposits = []
  total_collects = []
  all_grads_list = []
  with Pool(3) as pool:
    for grads_list, d, c in pool.imap_unordered(compute_grad, gen_args(replay_jsons)):
      all_grads_list.extend(grads_list)
      total_deposits = np.concatenate([total_deposits, d])
      total_collects = np.concatenate([total_collects, c])

  def apply_grad(grads_list):
    import train
    trainer = train.Trainer(None, model_dir)
    for i, grads in enumerate(grads_list):
      trainer.apply_grad(grads)
      print("apply grad at %s" % i)
    trainer.on_batch_finished()
    return trainer.checkpoint.step

  random.shuffle(all_grads_list)
  step = apply_grad(all_grads_list)

  D = np.mean(total_deposits)
  C = np.mean(total_collects)
  log_txt = ("****Step=%s, Avg deposite = %.3f, avg collect = %.3f, ratio=%.5f"
             % (int(step), D, C, (D / (C  + 0.001))))
  print(log_txt)

  log_path = os.path.join(model_dir, 'log.txt')
  with open(log_path, 'a') as f:
    f.write(log_txt + "\n")


def run_train(episode_dir, model_dir):
  replays = list(scan_for_replays(episode_dir))
  train_on_replays_multiprocessing(model_dir, replays)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  args = parser.parse_args()
  run_train(args.episode_dir, args.model_dir)


if __name__ == "__main__":
  main()

