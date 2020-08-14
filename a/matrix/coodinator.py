import os
import json
import time
import subprocess
import datetime
import argparse

import uuid


EPSILON_DECAY = 0.995


def run(episode_dir, model_dir, batch, epsilon, episode_steps, batch_size, args):

  for b in range(batch):
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(episode_dir, time_tag)

    subprocess.check_call(["python3", "lsd.py", "-o", batch_dir,
                           '-m', model_dir,
                           '--epsilon', str(epsilon),
                           '--episode_steps', str(episode_steps),
                           '--batch_size', str(batch_size),
                           '--num_players', str(args.num_players)])
    print("All episodes generation finished: ", batch_dir)

    subprocess.check_call(["python3", "batch_train.py", "-e", batch_dir,
                           '-m', model_dir])
    print("One training batch finished: ", batch_dir)
    print("Step: %s, epsilon value: %s" % (b, epsilon))

    epsilon *= EPSILON_DECAY



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
  run(args.episode_dir, args.model_dir, args.batch,
      args.epsilon, args.episode_steps, args.batch_size, args)


if __name__ == "__main__":
  main()
