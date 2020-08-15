
import argparse
import os
import uuid
import json
import datetime
from multiprocessing import Pool

from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

BATCH_SIZE = 6
EPISODE_STEPS = 10
BOARD_SIZE = 9

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def simulate(args):
  output_dir, model_dir, epsilon, cmd_args = args

  from train import Trainer
  import matrix_v0 as mat
  trainer = Trainer(mat.get_model(), model_dir)
  mat.STRATEGY = mat.ShipStrategy(trainer.model, epsilon)

  env = make("halite", {'episodeSteps': EPISODE_STEPS,
                        'size': BOARD_SIZE}, debug=True)
  env.run([mat.agent] * cmd_args.num_players)

  replay_json = env.toJSON()
  replay_id = replay_json['id']
  output_path = os.path.join(output_dir, replay_id + ".json")
  print('Output episode:', output_path)
  with open(output_path, 'w') as f:
      f.write(json.dumps(replay_json))


def run_lsd(output_dir, model_dir, epsilon, args):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  def gen_simulation_args():
    for i in range(BATCH_SIZE):
      yield output_dir, model_dir, epsilon, args

  def gen_simulations():
    sim_args = list(gen_simulation_args())
    with Pool() as pool:
      for replay_json in pool.imap_unordered(simulate, sim_args):
        pass

  gen_simulations()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--output_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('--epsilon', type=float, required=True)
  parser.add_argument('--episode_steps', type=int, default=400)
  parser.add_argument('--batch_size', type=int, default=24)
  parser.add_argument('--num_players', type=int, default=4)
  args = parser.parse_args()

  global EPISODE_STEPS, BATCH_SIZE
  EPISODE_STEPS = args.episode_steps
  BATCH_SIZE = args.batch_size
  run_lsd(args.output_dir, args.model_dir, args.epsilon, args)


if __name__ == "__main__":
  main()
