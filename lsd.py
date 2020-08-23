
import random
import argparse
import os
import json
from multiprocessing import Pool

from kaggle_environments import make

OUTPUT_DIR = None

work_dir = "/home/wangfei/repo/flynn/kaggle_halite4/"

def agent(name, collection=False, check=True):
  path = os.path.join(work_dir, collection and 'collection' or '', name)
  if check:
    assert os.path.exists(path), path
  return path


AGENTS = [
  agent("agent_tom_v1_0_0.py"),
  agent("agent_bee_v4_1_1.py", check=True),
  agent("agent_bee_v4_2_1.py", check=True),
  # agent("agent_bee_v4_8_3.py", check=True),
  # agent("agent_bee_v4_9_0.py", check=True),
  # agent("agent_bee_v4_9_1.py", check=True),
  # agent("agent_bee_v4_9_3.py", check=True) ,
  # agent("agent_bee_v4_9_4.py", check=True) ,

  agent("agent_bee_v4_9_5.py", check=True) ,
]


def simulate(output_dir):
  env = make("halite", {'episodeSteps': 400}, debug=True)

  agents = [a for a in AGENTS]
  random.shuffle(agents)
  names = [os.path.basename(a) for a in agents]

  env.run(agents)
  replay_json = env.toJSON()
  replay_json['agent_names'] = names

  replay_id = replay_json['id']
  rewards = replay_json['rewards']
  output_path = os.path.join(output_dir, replay_id + ".json")
  print(f"Episode {replay_id}: {names[0]}={rewards[0]} {names[1]}={rewards[1]}"
        f" :{names[2]}={rewards[2]} {names[3]}={rewards[3]}")
  with open(output_path, 'w') as f:
      f.write(json.dumps(replay_json))


def run_lsd(output_dir, episode_num):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  def gen_simulation_args():
    for i in range(episode_num):
      yield output_dir

  sim_args = list(gen_simulation_args())
  with Pool() as pool:
    for _ in pool.imap_unordered(simulate, sim_args):
      pass


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--output_dir', required=True)
  parser.add_argument('-c', '--episode_num', required=True, type=int)
  args = parser.parse_args()
  run_lsd(args.output_dir, args.episode_num)


if __name__ == "__main__":
  main()
