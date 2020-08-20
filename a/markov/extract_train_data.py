
import os
import json
import random
from pathlib import Path

import numpy as np
from numpy import savez_compressed
from kaggle_environments.envs.halite.helpers import *

import behaviour_model
import model_input
import replayer
from behaviour_model import INPUT_SHAPE, NUM_SHIP_ACTIONS
from model_input import MODEL_INPUT_SIZE, SHIP_ACTIONS


ACTION_TO_INDEX = {a:i for i, a in enumerate(SHIP_ACTIONS)}


SAMPLE_NUM = 3
DATA_DIR = "/home/wangfei/data/20200801_halite/scraping_data"

def get_json_file_paths():
  for path in Path(DATA_DIR).rglob('*.json'):
    if '_info' not in path.name and 'dead' not in str(path):
      yield str(path)


def is_player_eliminated(player, spawn_cost):
  """A player is eliminated when no ships left and no shipyard for building a
  ship, or not enough money to build one."""
  return (len(player.ship_ids) == 0 and
          (len(player.shipyard_ids) == 0 or player.halite < spawn_cost))

def get_info_path(replay_path):
  return replay_path[:-5] + "_info.json"

def gen_data(output_dir, replay_path, move_axis=True):
  parse_error = False
  try:
    with open(replay_path, 'r') as f:
      replay_json = json.loads(f.read())

    with open(get_info_path(replay_path), 'r') as f:
      info_json = json.loads(f.read())
  except Exception as e:
    print(e, replay_path)
    parse_error = True
    return 0

  if parse_error or len(replay_json['steps']) <= 30:
    path = os.path.join("/home/wangfei/data/20200801_halite/scraping_data/dead",
                        os.path.basename(replay_path))
    os.rename(replay_path, path)
    return 0

  X = []
  Y = []
  c = 0
  rep = replayer.Replayer(replay_json, player_id=0)
  for i in range(rep.total_steps):
    board = rep.get_board(i)
    for player in board.players.values():
      player_id = player.id
      if is_player_eliminated(player, board.configuration.spawn_cost):
        continue

      ships = [s for s in player.ships if s.next_action != ShipAction.CONVERT]

      # If there is no ship at this round
      if len(ships) == 0:
        continue
      num_sampled = min(SAMPLE_NUM, len(ships))
      c += num_sampled

      ships = random.sample(ships, num_sampled)
      mi = model_input.ModelInput(board, player_id)
      player_input = mi.get_player_input(move_axis=False)
      for ship in ships:
        x = mi.ship_centered_input(ship, player_input, move_axis=move_axis)
        X.append(x)

        y = ACTION_TO_INDEX[ship.next_action]
        Y.append(y)

  X = np.stack(X)
  Y = np.stack(Y)

  # Shuffle X and Y with a same index.
  s = np.arange(X.shape[0])
  np.random.shuffle(s)
  X = X[s]
  Y = Y[s]

  episode_id = os.path.basename(replay_path)[:-5]
  path = os.path.join(output_dir, f'{episode_id}.npz')
  np.savez(path, X=X, Y=Y)
  return c


def dry_run(output_dir, replay_paths):
  from multiprocessing import Pool
  from tqdm import tqdm
  from functools import partial
  total = 0

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  with Pool() as pool:
    gen = partial(gen_data, output_dir)
    for i in pool.imap_unordered(gen, tqdm(replay_paths)):
      total += i

  print("total = ", total)


# OUTPUT_DIR = "/home/wangfei/data/20200801_halite/train_data/X_valid_small"


replay_paths = list(get_json_file_paths())
random.shuffle(replay_paths)

TRAIN_DIR = "/home/wangfei/data/20200801_halite/train_data/X_train_median"
VALID_DIR = "/home/wangfei/data/20200801_halite/train_data/X_valid_median"
valid_replay_paths = replay_paths[-500:]
train_replay_paths = replay_paths[:2000]

# TRAIN_DIR = "/home/wangfei/data/20200801_halite/train_data/X_train_large"
# VALID_DIR = "/home/wangfei/data/20200801_halite/train_data/X_valid_large"
# valid_replay_paths = replay_paths[-500:]
# train_replay_paths = replay_paths[:-500]

# dry_run(VALID_DIR, valid_replay_paths)
dry_run(TRAIN_DIR, train_replay_paths)
