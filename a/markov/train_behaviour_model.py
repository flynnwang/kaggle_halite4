
import os
import json
import random
from pathlib import Path


import numpy as np
import keras
from tqdm import tqdm
from kaggle_environments.envs.halite.helpers import *

import behaviour_model
import model_input
import replayer
from behaviour_model import INPUT_SHAPE, NUM_SHIP_ACTIONS
from model_input import MODEL_INPUT_SIZE, SHIP_ACTIONS

ACTION_TO_INDEX = {a:i for i, a in enumerate(SHIP_ACTIONS)}

SAMPLE_NUM = 2
DATA_DIR = "/home/wangfei/data/20200801_halite/scraping_data"

# TRAIN_DIR = "/home/wangfei/data/20200801_halite/train_data/X_train_small"
# VALID_DIR = "/home/wangfei/data/20200801_halite/train_data/X_valid_small"
TRAIN_DIR = "/home/wangfei/data/20200801_halite/train_data/X_train_median"
VALID_DIR = "/home/wangfei/data/20200801_halite/train_data/X_valid_median"

# TRAIN_DIR = "/home/wangfei/data/20200801_halite/train_data/X_train_large"
# VALID_DIR = "/home/wangfei/data/20200801_halite/train_data/X_valid_large"


def get_json_file_paths():
  for path in Path(DATA_DIR).rglob('*.json'):
    if '_info' not in path.name and 'dead' not in str(path):
      yield str(path)


def is_player_eliminated(player, spawn_cost):
  """A player is eliminated when no ships left and no shipyard for building a
  ship, or not enough money to build one."""
  return (len(player.ship_ids) == 0 and
          (len(player.shipyard_ids) == 0 or player.halite < spawn_cost))


def gen_data(replay_path, move_axis=True):
  with open(replay_path, 'r') as f:
    replay_json = json.loads(f.read())

  total_steps = len(replay_json['steps'])
  if total_steps <= 30:
    return 0

  X = []
  Y = []
  c = 0
  rep = replayer.Replayer(replay_json, player_id=0)

  BOARD_START = 5
  prev_board = rep.get_board(BOARD_START-1)
  for i in range(BOARD_START, rep.total_steps):
    board = rep.get_board(i)
    for player_id, player in enumerate(board.players.values()):
      if is_player_eliminated(player, board.configuration.spawn_cost):
        continue

      ships = [s for s in player.ships if s.next_action != ShipAction.CONVERT]

      # If there is no ship at this round
      if len(ships) == 0:
        continue
      num_sampled = min(SAMPLE_NUM, len(ships))
      c += num_sampled

      ships = random.sample(ships, num_sampled)
      mi = model_input.ModelInput(board, player_id, prev_board=prev_board)
      player_input = mi.get_player_input(move_axis=False)
      for ship in ships:
        x = mi.ship_centered_input(ship, player_input, move_axis=move_axis)
        X.append(x)

        y = ACTION_TO_INDEX[ship.next_action]
        Y.append(y)
    prev_board = board

  X = np.stack(X)
  Y = np.stack(Y)

  # Shuffle X and Y with a same index.
  s = np.arange(X.shape[0])
  np.random.shuffle(s)
  X = X[s]
  Y = Y[s]
  return X, Y


def load_data(file_paths):
  X = []
  Y = []
  for path in file_paths:
    with np.load(path) as f:
      X.append(f['X'])
      Y.append(f['Y'])
  X = np.concatenate(X)
  Y = np.concatenate(Y)
  return X, Y


def list_all_files(data_dir):
  for f in tqdm(os.listdir(data_dir)):
    if not f.endswith("npz"):
      continue
    path = os.path.join(data_dir, f)
    yield path


def load_all_data(data_dir):
  paths = list(list_all_files(data_dir))
  return load_data(paths)


class DataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, file_paths, use_cached_data=False):
    'Initialization'
    self.file_paths = file_paths
    self.use_cached_data = use_cached_data
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return len(self.file_paths)

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    file_path = self.file_paths[index]
    if self.use_cached_data:
      X, Y = load_data([file_path])
    else:
      X, Y = gen_data(file_path)
    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    random.shuffle(self.file_paths)


model = behaviour_model.get_keras_unet()
optimizer = keras.optimizers.Adam(learning_rate=3e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])


MODEL_PATH = "/home/wangfei/data/20200801_halite/model/behaviour_model/model_15x15_0823.h5"
callbacks = [keras.callbacks.ModelCheckpoint(MODEL_PATH,
                                             save_weights_only=True,
                                             mode='min',
                                             monitor='val_loss',
                                             save_freq=1000)]

# OLD_MODEL_PATH = "/home/wangfei/data/20200801_halite/model/behaviour_model/test_model_v3.h5"
# model.load_weights(OLD_MODEL_PATH)

epochs = 100

# train_gen = DataGenerator(list(list_all_files(TRAIN_DIR)), use_cached_data=True)
# valid_gen = DataGenerator(list(list_all_files(VALID_DIR)), use_cached_data=True)

replay_paths = list(get_json_file_paths())
random.shuffle(replay_paths)
valid_replay_paths = replay_paths[-500:]
train_replay_paths = replay_paths[:-500]

# valid_replay_paths = replay_paths[-50:]
# train_replay_paths = replay_paths[:500]
train_gen = DataGenerator(train_replay_paths)
valid_gen = DataGenerator(valid_replay_paths)

model.fit(train_gen,
          validation_data=valid_gen,
          epochs=epochs,
          callbacks=callbacks)
