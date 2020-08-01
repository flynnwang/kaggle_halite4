
import argparse
import os
import uuid
import json
from multiprocessing import Pool
import datetime

from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

from matrix_v0 import agent, get_model
from train import Trainer, gen_replays, train_on_replays


def simulate(output_path):
  env = make("halite", {'episodeSteps': 400}, debug=True)
  # TODO: why there ase multi-processes running?
  env.run([agent] * 4)

  print('Output episode:', output_path)
  with open(output_path, 'w') as f:
      f.write(json.dumps(env.toJSON()))


def run_lsd(output_dir, times):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  # for i in range(times):
    # time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # episode_name = '%s_%s.json' % (time_tag, str(uuid.uuid4())[:8])
    # episode_path = os.path.join(output_dir, episode_name)
    # simulate(episode_path)

  trainer = Trainer(get_model())
  train_on_replays(trainer, gen_replays(output_dir))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--agent')
  parser.add_argument('-o', '--output_dir', required=True)
  parser.add_argument('-t', '--times', required=True, type=int)
  args = parser.parse_args()
  run_lsd(args.output_dir, args.times)


if __name__ == "__main__":
  main()
