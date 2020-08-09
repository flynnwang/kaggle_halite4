
import argparse
import os
import uuid
import json
import datetime
from multiprocessing import Pool

from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

BATCH_SIZE = 6
EPISODE_STEPS = 100

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def simulate(args):
  output_path, model_dir, epsilon = args

  from train import Trainer
  import matrix_v0 as mat
  trainer = Trainer(mat.get_model(), model_dir)
  mat.STRATEGY = mat.ShipStrategy(trainer.model, epsilon)

  env = make("halite", {'episodeSteps': EPISODE_STEPS}, debug=True)
  env.run([mat.agent] * 4)

  print('Output episode:', output_path)
  replay_json = env.toJSON()
  tmp_path = output_path + '.tmp'
  with open(tmp_path, 'w') as f:
      f.write(json.dumps(replay_json))

  os.rename(tmp_path, output_path)


def run_lsd(output_dir, model_dir, epsilon):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  def gen_simulation_args():
    for i in range(BATCH_SIZE):
      time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      episode_name = '%s_%s.json' % (time_tag, str(uuid.uuid4())[:8])
      episode_path = os.path.join(output_dir, episode_name)
      yield episode_path, model_dir, epsilon

  def gen_simulations():
    sim_args = list(gen_simulation_args())
    with Pool(processes=6) as pool:
      for replay_json in pool.imap_unordered(simulate, sim_args):
        pass

  gen_simulations()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--output_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('--epsilon', type=float, required=True)
  args = parser.parse_args()
  run_lsd(args.output_dir, args.model_dir, args.epsilon)


if __name__ == "__main__":
  main()
