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

MIN_WEIGHT = -99999

BOARD_SIZE = 21
INPUT_MAP_SIZE = (BOARD_SIZE, BOARD_SIZE)
HALITE_NORMALIZTION_VAL = 500.0
TOTAL_STEPS = 400

SHIPYARD_ACTIONS = list(ShipyardAction) + [None]
NUM_SHIPYARD_ACTIONS = len(SHIPYARD_ACTIONS)

SHIP_ACTIONS = [a for a in list(ShipAction) if a != ShipAction.CONVERT] + [None]
NUM_SHIP_ACTIONS = len(SHIP_ACTIONS)


# MODEL_PATH = '/data/wangfei/data/202007_halite/unet.h5'
MODEL_PATH = '/home/wangfei/data/20200801_halite/model/unet.h5'


def get_model2(input_shape=(BOARD_SIZE, BOARD_SIZE, 7),
              num_ship_actions=NUM_SHIP_ACTIONS,
              num_shipyard_actions=NUM_SHIPYARD_ACTIONS,
              input_padding=((5, 6), (5, 6))):
  from keras import layers
  from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten
  from keras.regularizers import l2

  inputs = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(input_padding)(inputs)
  x = layers.Conv2D(32, 1, strides=1, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  conv1 = Conv2D(64,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(x)
  conv1 = Conv2D(64,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv1)
  conv1 = layers.BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(128,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv2)
  conv2 = layers.BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(256,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv3)
  conv3 = layers.BatchNormalization()(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  def decoder(input_tensor):
    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2D(256,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(
                     UpSampling2D(size=(2, 2))(input_tensor))

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    return layers.Cropping2D(input_padding)(conv9)

  ship_outputs = Conv2D(num_ship_actions, 3, activation="softmax", padding="same",
                        kernel_initializer = 'he_normal')(decoder(pool3))
  critic_outputs = Conv2D(1, 3, activation="linear",
                          kernel_regularizer=l2(1e-4),
                          padding="same")(decoder(pool3))

  model = keras.Model(inputs, outputs=[ship_outputs, critic_outputs])
  return model


def get_model(input_shape=(BOARD_SIZE, BOARD_SIZE, 7),
              num_ship_actions=NUM_SHIP_ACTIONS,
              num_shipyard_actions=NUM_SHIPYARD_ACTIONS):
  import keras
  from keras import layers
  from keras.regularizers import l2

  inputs = layers.Input(shape=input_shape)

  # ((top_pad, bottom_pad), (left_pad, right_pad))
  input_padding = ((5, 6), (5, 6))
  x = layers.ZeroPadding2D(input_padding)(inputs)

  ### [First half of the network: downsampling inputs] ###

  # Entry block
  x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
      previous_block_activation
    )
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  ### [Second half of the network: upsampling inputs] ###

  def decoder(x):
    previous_block_activation = x  # Set aside residual
    for filters in [256, 128, 64, 32]:
      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.UpSampling2D(2)(x)

      # Project residual
      residual = layers.UpSampling2D(2)(previous_block_activation)
      residual = layers.Conv2D(filters, 1, padding="same")(residual)
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual
    return layers.Cropping2D(input_padding)(x)

  encoder_end = x
  ship_outputs = layers.Conv2D(num_ship_actions, 3, activation="softmax",
                        padding="same")(decoder(encoder_end))
  critic_outputs = layers.Conv2D(1, 1,
                          activation="linear",
                          kernel_regularizer=l2(1e-4),
                          bias_regularizer=l2(1e-4),
                          padding="same")(decoder(encoder_end))
  # Define the model
  model = keras.Model(inputs, outputs=[ship_outputs, critic_outputs])
  return model


def cargo(player):
  """Computes the cargo value for a player."""
  return sum([s.halite for s in player.ships], 0)


class ModelInput:
  """Convert a board state into 32x32x6 of model input.
  Every map is origined at (5, 5)

  1. halite map
  2. a map of ships
  3. a map of shipyards
  4. auxilary infomation: step progress, my halite, my cargo, my ships, and values / total.
  """

  def __init__(self, board):
    self.board = board

  def get_input(self, move_axis=True):
    halites = self.halite_cell_map()

    me = self.board.current_player
    ship_position_map = self.player_ship_map(me.id, halite_layer=False)
    ship_cargo_map = self.player_ship_map(me.id, halite_layer=True)
    shipyard_map = self.player_shipyard_map(me.id)

    enemy_position_map = self.enemy_ship_map(halite_layer=False)
    enemy_cargo_map = self.enemy_ship_map(halite_layer=True)
    enemy_shipyard_map = self.enemy_shipyard_map()
    # aux_map = self.get_auxiliary_map()

    maps = [halites, ship_position_map, ship_cargo_map, shipyard_map,
            enemy_position_map, enemy_cargo_map, enemy_shipyard_map]
    v = np.stack(maps)
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

  def player_ship_map(self, player_id, halite_layer=True):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for ship in self.board.players[player_id].ships:
      position = ship.position
      value = ship.halite if halite_layer else 1
      v[position.x, position.y] = value
    return v

  def player_shipyard_map(self, player_id):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for yard in self.board.players[player_id].shipyards:
      position = yard.position
      v[position.x, position.y] = 1
    return v

  def enemy_ship_map(self, player_id=None, halite_layer=True):
    current_player_id = self.board.current_player.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.opponents:
      if player_id != None and player.id != player_id:
        continue
      v += self.player_ship_map(player_id=player.id, halite_layer=halite_layer)
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


class ShipStrategy(StrategyBase):

  def __init__(self, model=None, epsilon=0):
    super().__init__()
    self.model = model
    if self.model is None:
      self.model = get_model()
      assert os.path.exists(MODEL_PATH)
      self.model.load_weights(MODEL_PATH)
    self.epsilon = epsilon

  def assign_unti_actions(self, units, unit_probs, actions):
    for unit in units:
      # skip if already has an action.
      if unit.next_action:
        continue

      position = unit.position
      unit_action_probs = unit_probs[0][position.x, position.y, :]

      # Eposilon exploration.
      # sample_probs = 0.9 * unit_action_probs + 0.1 * (np.ones(len(actions)) / len(actions))
      # sample_probs = sample_probs / np.sum(sample_probs)

      if random.random() < self.epsilon:
        unit_action_probs = np.ones(len(actions)) / len(actions)

      # TODO(wangfei): use maximize when deploy
      action_idx = np.random.choice(len(actions), p=unit_action_probs)
      unit.next_action = actions[action_idx]

  def convert_shipyard_if_none(self):
    if not self.me.ship_ids or len(self.me.shipyard_ids) > 0:
      return

    ships = self.me.ships
    random.shuffle(ships)
    for ship in ships:
      if ship.halite + self.me.halite >= self.c.convert_cost:
        ship.next_action = ShipAction.CONVERT
        break

  def spawn_ships(self):
    # TODO(wangfei): stop spawn when step > XXX
    halite = self.me.halite
    for yard in self.me.shipyards:
      if halite >= self.c.spawn_cost:
        yard.next_action = ShipyardAction.SPAWN
        halite -= self.c.spawn_cost

  def compute_ship_moves(self):
    """Computes ship moves to avoid collision.

    Maximize total expected value.
    * prefer the move output from model
    * avoid collision
    """
    board_size = self.c.size

    def compute_weight(ship, next_position, expect_position):
      wt = random.random()
      # Prefer the model predicted position.
      if next_position == expect_position:
        wt += 10

      # In case the expect position is blocked, perfer stay.
      if next_position == ship.position:
        wt += 1

      # Do not move to a spawning cell.
      next_cell = self.board[next_position]
      yard = next_cell.shipyard
      if (yard and yard.player_id == self.me.id and
          yard.next_action == ShipyardAction.SPAWN):
        wt = MIN_WEIGHT
      return wt

    def make_move(position, action):
      next_position = None
      if action is None:
        next_position = position
      elif action == ShipAction.NORTH:
        next_position = position + Point(0, +1)
      elif action == ShipAction.SOUTH:
        next_position = position + Point(0, -1)
      elif action == ShipAction.EAST:
        next_position = position + Point(-1, 0)
      elif action == ShipAction.WEST:
        next_position = position + Point(+1, 0)
      else:
        assert False, "no action found for %s" % action
      return next_position % board_size

    def next_position_ship_action(position, next_position):
      if position == next_position:
        return None
      if (position + Point(0, 1)) % board_size == next_position:
        return ShipAction.NORTH
      if (position + Point(1, 0)) % board_size == next_position:
        return ShipAction.EAST
      if (position + Point(0, -1)) % board_size == next_position:
        return ShipAction.SOUTH
      if (position + Point(-1, 0)) % board_size == next_position:
        return ShipAction.WEST
      assert False, '%s, %s' % (position, next_position)

    # Skip only convert ships.
    ships = [s for s in self.ships if s.next_action != ShipAction.CONVERT]
    next_positions = {
        make_move(s.position, move) for s in ships for move in SHIP_ACTIONS
    }

    position_to_index = {pos: i for i, pos in enumerate(next_positions)}
    C = np.ones((len(ships), len(next_positions))) * MIN_WEIGHT
    for ship_idx, ship in enumerate(ships):
      expect_position = make_move(ship.position, ship.next_action)
      for move in SHIP_ACTIONS:
        next_position = make_move(ship.position, move)
        poi_idx = position_to_index[next_position]
        C[ship_idx, poi_idx] = compute_weight(ship, next_position, expect_position)

    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)
    index_to_position = list(next_positions)
    changed_actions = 0
    for ship_idx, poi_idx in zip(rows, cols):
      ship = ships[ship_idx]
      next_position = index_to_position[poi_idx]
      final_action = next_position_ship_action(ship.position, next_position)
      if final_action != ship.next_action:
        changed_actions += 1
        # print("ship(id=%s at %s), change action from %s to %s."
              # % (ship.id, ship.position, ship.next_action, final_action))
      ship.next_action = final_action
      # print(ship.id, 'at', ship.position, 'goto', next_position)

    assert len(rows) == len(ships), "match=%s, ships=%s" % (len(rows), len(ships))

  def execute(self):
    self.convert_shipyard_if_none()
    self.spawn_ships()

    model_input = ModelInput(self.board)
    input = model_input.get_input(move_axis=True)
    ship_probs, _ = self.model.predict(np.expand_dims(input, axis=0))
    self.assign_unti_actions(self.me.ships, ship_probs, SHIP_ACTIONS)
    self.compute_ship_moves()


STRATEGY = None

@board_agent
def agent(board):
  global STRATEGY

  if STRATEGY is None:
    STRATEGY = ShipStrategy()
  STRATEGY.update(board)
  STRATEGY.execute()
