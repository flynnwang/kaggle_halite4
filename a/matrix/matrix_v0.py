# -*- coding: utf-8 -*-

import os
import random
import timeit
import logging
from collections import Counter
from enum import Enum, auto

import numpy as np
import scipy.optimize
from scipy.ndimage.interpolation import shift
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import evaluate, make

import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten


OFFSET = Point(5, 5)

BOARD_SIZE = 21
INPUT_MAP_SIZE = (32, 32)
HALITE_NORMALIZTION_VAL = 500.0
TOTAL_STEPS = 400

# num_classes = 4 move + 1 stay + 1 convert + 1 spawn
def get_model(input_shape=(32, 32, 6), num_ship_actions=6, num_shipyard_actions=1):
  inputs = Input(shape=input_shape)

  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  drop3 = Dropout(0.5)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
  # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  # drop4 = Dropout(0.5)(conv4)
  # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  # drop5 = Dropout(0.5)(conv5)

  # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  # merge6 = concatenate([drop4,up6], axis = 3)
  # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  def decoder(input_tensor):
    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(input_tensor))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    return conv9

  # Add a per-pixel classification layer
  # action_decoder = decoder(pool3)
  ship_outputs = Conv2D(num_ship_actions, 3, activation="softmax", padding="same", kernel_initializer = 'he_normal')(decoder(pool3))
  shipyard_outputs = Conv2D(num_shipyard_actions, 3, activation="sigmoid", padding="same", kernel_initializer='he_normal')(decoder(pool3))

  critic_pool = decoder(pool3)
  critic_flattend = Flatten()(critic_pool)
  critic_dennse = tf.keras.layers.Dense(32, activation='relu')(critic_flattend)
  critic_dennse = tf.keras.layers.Dense(32, activation='relu')(critic_dennse)
  critic_outputs = tf.keras.layers.Dense(1)(critic_dennse)

  # Define the model
  model = keras.Model(inputs, outputs=[ship_outputs, shipyard_outputs, critic_outputs])
  # model = keras.Model(inputs, outputs=[ship_outputs, shipyard_outputs])
  return model


def cargo(player):
  """Computes the cargo value for a player."""
  return sum([s.halite for s in player.ships], 0)

class ModelInput:
  """Convert a board state into 32x32x6 of model input.
  Every map is origined at (5, 5)

  1. halite map
  2. a map of ship of mine
  3. a map of shipyard of mine
  4. a enemy ship map
  5. a enemy shipyard map
  6. auxilary infomation: step progress, my halite, my cargo, my ships, and values / total.
  """

  def __init__(self, board):
    self.board = board

  def get_input(self, move_axis=True):
    halites = self.halite_cell_map()

    me = self.board.current_player
    my_ship_map = self.player_ship_map(player_id=me.id)
    my_shipyard_map = self.player_shipyard_map(player_id=me.id)

    enemy_ship_map = self.enemy_ship_map()
    enemy_shipyard_map = self.enemy_shipyard_map()

    aux_map = self.get_auxiliary_map()

    maps = [halites, my_ship_map, my_shipyard_map, enemy_ship_map,
            enemy_shipyard_map, aux_map]
    v = np.stack([shift(m, OFFSET) for m in maps])
    if move_axis:
      v = np.moveaxis(v, 0, -1)
    return v

  def get_auxiliary_map(self):
    v = np.zeros(shape=INPUT_MAP_SIZE)

    # border marks
    sz = BOARD_SIZE-1
    v[0, 0] = v[0, sz] = v[sz, 0] = v[sz, sz] = 1

    # progress
    v[0, 5] = (self.board.step + 1) / TOTAL_STEPS

    total_ships = sum(len(p.ship_ids) for p in self.board.players.values()) or 1
    total_halite = sum(p.halite for p in self.board.players.values()) or 1
    total_cargo = sum(cargo(p) for p in self.board.players.values()) or 1

    # my ships
    me = self.board.current_player
    v[5, 5] = len(me.ship_ids) / 50
    v[5, 10] = len(me.ship_ids) / total_ships

    # my halite
    v[10, 5] = me.halite / HALITE_NORMALIZTION_VAL
    v[10, 10] = me.halite / total_halite

    # my cargo
    v[15, 5] = cargo(me) / HALITE_NORMALIZTION_VAL
    v[15, 10] = cargo(me) / total_cargo
    return v

  def halite_cell_map(self):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for position, cell in self.board.cells.items():
      v[position.x, position.y] = cell.halite / HALITE_NORMALIZTION_VAL
    return v

  def player_ship_map(self, player_id):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for ship in self.board.players[player_id].ships:
      position = ship.position
      # v[position.x, position.y] = (ship.halite + 100) / HALITE_NORMALIZTION_VAL
      v[position.x, position.y] = (-1 if ship.halite == 0 else
                                   (ship.halite / HALITE_NORMALIZTION_VAL))
    return v

  def player_shipyard_map(self, player_id):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for yard in self.board.players[player_id].shipyards:
      position = yard.position
      v[position.x, position.y] = 1
    return v

  def enemy_ship_map(self, player_id=None):
    current_player_id = self.board.current_player.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.opponents:
      if player_id != None and player.id != player_id:
        continue
      v += self.player_ship_map(player_id=player.id)
    return v

  def enemy_shipyard_map(self, player_id=None):
    current_player_id = self.board.current_player.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.opponents:
      if player_id != None and player.id != player_id:
        continue
      v += self.player_shipyard_map(player_id=player.id)
    return v


class StrategyBase:
  """Class with board related method."""

  def __init__(self):
    super().__init__()
    self.board = None

  @property
  def me(self):
    return self.board.current_player

  @property
  def c(self):
    return self.board.configuration

  @property
  def sz(self):
    return self.c.size

  @property
  def step(self):
    return self.board.step

  @property
  def num_ships(self):
    return len(self.me.ship_ids)

  @property
  def num_shipyards(self):
    return len(self.me.shipyard_ids)

  @property
  def enemy_shipyards(self):
    for e in self.board.opponents:
      for y in e.shipyards:
        yield y

  @property
  def enemy_ships(self):
    for e in self.board.opponents:
      for s in e.ships:
        yield s

  def update(self, board):
    self.board = board

    # Cache it to eliminate repeated list constructor.
    self.shipyards = self.me.shipyards
    self.ships = self.me.ships

  def execute(self):
    pass

  def __call__(self):
    self.execute()


SHIPYARD_ACTIONS = list(ShipyardAction) + [None]
NUM_SHIPYARD_ACTIONS = len(SHIPYARD_ACTIONS)

SHIP_ACTIONS = list(ShipAction) + [None]
NUM_SHIP_ACTIONS = len(SHIP_ACTIONS)


# MODEL_PATH = '/data/wangfei/data/202007_halite/unet.h5'
MODEL_PATH = '/home/wangfei/data/20200801_halite/model/unet.h5'

class ShipStrategy(StrategyBase):

  def __init__(self, model=None):
    super().__init__()
    self.model = model
    if self.model is None:
      self.model = get_model()
      assert os.path.exists(MODEL_PATH)
      self.model.load_weights(MODEL_PATH)

  def assign_ship_actions(self, ship_out):
    for ship in self.ships:
      position = ship.position + OFFSET
      ship_action_prob = ship_out[0][position.x, position.y, :NUM_SHIP_ACTIONS]

      # TODO(wangfei): use maximize when deply
      action_idx = np.random.choice(NUM_SHIP_ACTIONS, p=ship_action_prob)
      ship.next_action = SHIP_ACTIONS[action_idx]

  def assign_shipyard_actions(self, yard_out):
    for yard in self.shipyards:
      position = yard.position + OFFSET
      prob = yard_out[0][position.x, position.y, 0]

      # TODO(wangfei): use maximize when deply
      action_idx = np.random.choice(NUM_SHIPYARD_ACTIONS, p=[prob, 1-prob])
      yard.next_action = SHIPYARD_ACTIONS[action_idx]

  def execute(self):
    model_input = ModelInput(self.board)
    input = model_input.get_input(move_axis=True)
    ship_out, yard_out, _ = self.model.predict(np.expand_dims(input, axis=0))
    self.assign_ship_actions(ship_out)
    self.assign_shipyard_actions(yard_out)


STRATEGY = None

@board_agent
def agent(board):
  global STRATEGY

  if STRATEGY is None:
    STRATEGY = ShipStrategy()
  STRATEGY.update(board)
  STRATEGY.execute()
