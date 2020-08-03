import os
import json
import time
import subprocess
import datetime
import argparse

import uuid

def run(episode_dir, model_dir, batch):

  for b in range(batch):
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(episode_dir, time_tag)

    lsd_proc = subprocess.Popen(["python3", "lsd.py", "-o", batch_dir, '-m', model_dir])
    lsd_proc.wait()
    print("All episodes generation finished: ", batch_dir)

    train_proc = subprocess.Popen(["python3", "batch_train.py", "-e", batch_dir, '-m', model_dir])
    train_proc.wait()
    print("One training batch finished: ", batch_dir)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('-b', '--batch', required=True, type=int)
  args = parser.parse_args()
  run(args.episode_dir, args.model_dir, args.batch)


if __name__ == "__main__":
  main()
