
import argparse
import os
import uuid
import json
from multiprocessing import Pool

from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *


def simulate(agent_path,  output_path):
  env = make("halite", {'episodeSteps': 400}, debug=True)
  env.run([agent_path] * 4)
  with open(output_path, 'w') as f:
      f.write(json.dumps(env.toJSON()))

def run_lsd(agent_path, output_dir, times):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  with Pool() as pool:
    results = []
    for i in range(times):
      episode_name = '%s_%s.json' % (i, str(uuid.uuid4())[:6])
      episode_path = os.path.join(output_dir, episode_name)
      r = pool.apply_async(simulate, (agent_path, episode_path))
      results.append(r)

    for r in results:
      r.get()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--agent', required=True)
  parser.add_argument('-o', '--output_dir', required=True)
  parser.add_argument('-t', '--times', required=True, type=int)
  args = parser.parse_args()
  run_lsd(args.agent, args.output_dir, args.times)


if __name__ == "__main__":
  main()
