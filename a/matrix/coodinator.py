import os
import json
import time
import subprocess
import datetime
import argparse

import uuid

def run(episode_dir, model_dir, batch):

  # epsilon = 1.0
  epsilon = 0.05
  epsilon_decay = 0.995

  for b in range(batch):
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(episode_dir, time_tag)

    subprocess.check_call(["python3", "lsd.py", "-o", batch_dir, '-m', model_dir,
                           '--epsilon', str(epsilon)])
    print("All episodes generation finished: ", batch_dir)

    subprocess.check_call(["python3", "batch_train.py", "-e", batch_dir, '-m', model_dir])
    print("One training batch finished: ", batch_dir)
    print("Step: %s, epsilon value: %s" % (b, epsilon))

    epsilon *= epsilon_decay



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('-b', '--batch', required=True, type=int)
  args = parser.parse_args()
  run(args.episode_dir, args.model_dir, args.batch)


if __name__ == "__main__":
  main()
