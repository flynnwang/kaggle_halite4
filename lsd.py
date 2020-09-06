
import random
import argparse
import os, sys
import json
from multiprocessing import Pool
import getpass

from kaggle_environments import make

OUTPUT_DIR = None

username = getpass.getuser()
sys.path.insert(0, f"/home/wangfei/repo/{username}/kaggle_halite4")
sys.path.insert(0, f"/home/wangfei/repo/{username}/kaggle_halite4/a/markov")


work_dir = f"/home/{username}/repo/flynn/kaggle_halite4/"

def agent(name, collection=False, check=True):
  path = os.path.join(work_dir, collection and 'collection' or '', name)
  if check:
    assert os.path.exists(path), path
  return path


AGENTS = [
  agent("agent_tom_v1_0_0.py"),
  agent("agent_bee_v4_1_1.py", check=True),
  # agent("optimus_mining_agent.py", collection=True),
  agent("agent_bee_v4_2_1.py", check=True),

  agent("agent_bee_v4_13_2.py", check=True),
  # agent("agent_bee_v4_13_1.py", check=True),
  # agent("agent_bee_v4_13_0.py", check=True),
  # agent("agent_bee_v4_12_1.py", check=True),

  # agent("agent_bee_v4_11_21.py", check=True),
  # agent("agent_bee_v4_11_20.py", check=True),
  # agent("agent_bee_v4_11_19.py", check=True),
  # agent("agent_bee_v4_11_18.py", check=True),
  # agent("agent_bee_v4_11_17.py", check=True),


  # agent("agent_bee_v4_2_2.py", check=True),
  # agent("agent_bee_v4_8_3.py", check=True),
  # agent("agent_bee_v4_9_0.py", check=True),
  # agent("agent_bee_v4_9_1.py", check=True),
  # agent("agent_bee_v4_9_3.py", check=True) ,
  # agent("agent_bee_v4_9_4.py", check=True) ,

  # agent("agent_bee_v4_9_5.py", check=True) ,
  # agent("agent_bee_v4_9_6.py", check=True) ,
  # agent("agent_bee_v4_9_7.py", check=True) ,
  # agent("agent_bee_v4_9_8.py", check=True) ,
  # agent("agent_bee_v4_9_9.py", check=True) ,
  # agent("agent_bee_v4_9_10.py", check=True),
  # agent("agent_bee_v4_9_12.py", check=True),
  # agent("agent_bee_v4_9_13.py", check=True),
  # agent("agent_bee_v4_9_14.py", check=True),
  # agent("agent_bee_v4_9_15.py", check=True),
  # agent("agent_bee_v4_9_16.py", check=True),
  # agent("agent_bee_v4_9_17.py", check=True),
  # agent("agent_bee_v4_9_18.py", check=True),
  # agent("agent_bee_v4_9_19.py", check=True),
  # agent("agent_bee_v4_9_20.py", check=True),
  # agent("agent_bee_v4_9_21.py", check=True),
  # agent("agent_bee_v4_9_22.py", check=True),
  # agent("agent_bee_v4_9_23.py", check=True),
  # agent("agent_bee_v4_9_24.py", check=True),
  # agent("agent_bee_v4_9_25.py", check=True),
  # agent("agent_bee_v4_9_26.py", check=True),
  # agent("agent_bee_v4_9_27.py", check=True),
  # agent("agent_bee_v4_10_0.py", check=True),
  # agent("agent_bee_v4_11_0.py", check=True),
  # agent("agent_bee_v4_11_2.py", check=True),
  # agent("agent_bee_v4_11_3.py", check=True),
  # agent("agent_bee_v4_11_4.py", check=True),
  # agent("agent_bee_v4_11_6.py", check=True),
  # agent("agent_bee_v4_11_7.py", check=True),
  # agent("agent_bee_v4_11_8.py", check=True),
  # agent("agent_bee_v4_11_9.py", check=True),
  # agent("agent_bee_v4_11_10.py", check=True),
  # agent("agent_bee_v4_11_11.py", check=True),
  # agent("agent_bee_v4_11_12.py", check=True),
  # agent("agent_bee_v4_11_13.py", check=True),
  # agent("agent_bee_v4_11_16.py", check=True),

  # agent("agent_v5_0_0.py", check=True) ,
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
  print(f"Episode {replay_id.split('-')[0]} {names[0]}={rewards[0]} {names[1]}={rewards[1]}"
        f" :{names[2]}={rewards[2]} {names[3]}={rewards[3]}")
  with open(output_path, 'w') as f:
      f.write(json.dumps(replay_json))

  # Save only meta info for computing statistics.
  output_path = os.path.join(output_dir, replay_id + "_info.json")
  j = {
    'rewards': rewards,
    'agent_names': names
  }
  with open(output_path, 'w') as f:
      f.write(json.dumps(j))


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
