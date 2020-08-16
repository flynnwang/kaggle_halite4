import argparse
import os
import json
import subprocess
import datetime
import uuid
import random
from multiprocessing import Pool, Queue, Process

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BATCH_SIZE = 24
EPISODE_STEPS = 200

import train
import numpy as np


def compute_grad(args, trainer):
  replay_json, model_dir, norm_params = args

  num_players = len(replay_json['rewards'])
  player_ids = list(range(num_players))

  grads = []
  trainer.normalization_params = norm_params
  for player_id in player_ids:
    boards = list(train.gen_player_states(replay_json, player_id))
    g = trainer.train(boards, apply_grad=False)
    grads.extend(g)
  return grads


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


def replay_to_ship_rewards(args, trainer):
  replay_json, model_dir = args

  num_players = len(replay_json['rewards'])

  stats = np.zeros(2) # total deposit, total collect
  total_deposits = []
  total_collects = []
  total_advantages = []
  total_returns = []

  for player_id in range(num_players):
    boards = train.gen_player_states(replay_json, player_id)
    boards = list(boards)

    X = np.array([train.ModelInput(b).get_input() for b in boards],
                 dtype=np.float32)
    _, critic_values = trainer.model.predict(X)
    advs, returns, life_time = train.compute_ship_advantages(boards, critic_values)
    for ship_id, adv in advs.items():
      start, end = life_time[ship_id]
      total_advantages.extend(adv[start: end+1])

    for ship_id, rt in returns.items():
      start, end = life_time[ship_id]
      total_returns.extend(rt[start: end+1])

    b = boards[-1]
    total_deposits.append(b.total_deposite)
    total_collects.append(b.total_collect)

  max_reward = max(replay_json['rewards'])
  return total_deposits, total_collects, total_advantages, total_returns, max_reward


def get_normalization_params(model_dir, replays, trainer):
  def gen_args(replay_jsons):
    for replay_json in replay_jsons:
      yield replay_to_ship_rewards((replay_json, model_dir), trainer)

  total_deposits = []
  total_collects = []
  total_advantages = []
  total_returns = []
  max_rewards = []
  for d, c, a, r, mx_reward in gen_args(replays):
    total_deposits.extend(d)
    total_collects.extend(c)
    total_advantages.extend(a)
    total_returns.extend(r)
    max_rewards.append(mx_reward)

  D = np.mean(total_deposits)
  C = np.mean(total_collects)
  log_1 = ("****Avg deposite = %.3f, avg collect = %.3f, ratio=%.5f, max_reward=%.0f"
           % (D, C, (D / (C  +1.0)), np.mean(max_rewards)))

  a_mean, a_std = np.mean(total_advantages), np.std(total_advantages)
  log_2 = "****Stats of advantages: mean=%.5f, std=%.5f" % (a_mean, a_std)

  r_mean, r_std = np.mean(total_returns), np.std(total_returns)
  log_3 = "****Stats of returns: mean=%.5f, std=%.5f" % (r_mean, r_std)
  logs = [log_1, log_2, log_3]
  for l in logs:
    print(l)
  return (a_mean, a_std), logs


def train_on_replays(model_dir, replay_jsons, norm_params, log_lines, trainer):
  def gen_args(replay_jsons):
    for replay_json in replay_jsons:
      args = (replay_json, model_dir, norm_params)
      yield compute_grad(args, trainer)

  all_grads_list = []
  for grads_list in gen_args(replay_jsons):
    all_grads_list.extend(grads_list)

  def apply_grad(grads_list):
    trainer = train.Trainer(None, model_dir)
    for i, grads in enumerate(grads_list):
      trainer.apply_grad(grads)
      print("apply grad at %s" % i)
    trainer.on_batch_finished()
    return trainer.checkpoint.step

  random.shuffle(all_grads_list)
  step = apply_grad(all_grads_list)

  log_path = os.path.join(model_dir, 'log.txt')
  with open(log_path, 'a') as f:
    f.write("Step=%s\n" % int(step))
    for l in log_lines:
      print(l)
      f.write(l + '\n')


def run_train(episode_dir, model_dir, trainer):
  replays = list(scan_for_replays(episode_dir))
  norm_params, log_lines = get_normalization_params(model_dir, replays, trainer)
  train_on_replays(model_dir, replays, norm_params, log_lines, trainer)


def run(episode_dir, model_dir, batch, epsilon, args):
  epsilon_decay = 0.998

  trainer = train.Trainer(None, model_dir)
  for b in range(batch):
    print("Step: %s, epsilon value: %s" % (b, epsilon))

    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(episode_dir, time_tag)
    subprocess.check_call(["python3", "lsd.py", "-o", batch_dir,
                           '-m', model_dir,
                           '--epsilon', str(epsilon),
                           '--episode_steps', str(args.episode_steps),
                           '--batch_size', str(args.batch_size),
                           '--num_players', str(args.num_players)])
    print("All episodes generation finished: ", batch_dir)

    run_train(batch_dir, model_dir, trainer)
    trainer.on_batch_finished()




def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('-b', '--batch', required=True, type=int)
  parser.add_argument('--epsilon', type=float, default=1.0)
  parser.add_argument('--episode_steps', type=int, default=200)
  parser.add_argument('--batch_size', type=int, default=24)
  parser.add_argument('--num_players', type=int, default=4)
  args = parser.parse_args()

  global EPISODE_STEPS, BATCH_SIZE
  EPISODE_STEPS = args.episode_steps
  BATCH_SIZE = args.batch_size
  run(args.episode_dir, args.model_dir, args.batch, args.epsilon, args)


if __name__ == "__main__":
  main()
