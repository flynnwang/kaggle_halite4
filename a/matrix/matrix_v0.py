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

# BOARD_SIZE = 21
BOARD_SIZE = 9
INPUT_MAP_SIZE = (BOARD_SIZE, BOARD_SIZE)
HALITE_NORMALIZTION_VAL = 1000.0
TOTAL_STEPS = 400

SHIPYARD_ACTIONS = list(ShipyardAction) + [None]
NUM_SHIPYARD_ACTIONS = len(SHIPYARD_ACTIONS)

SHIP_ACTIONS = [a for a in list(ShipAction) if a != ShipAction.CONVERT] + [None]
NUM_SHIP_ACTIONS = len(SHIP_ACTIONS)


MODEL_PATH = '/home/wangfei/data/20200801_halite/model/unet.h5'

PADDING_LEFT_TOP = (32 - BOARD_SIZE) // 2
PADDING_RIGHT_BOTTOM = 32 - PADDING_LEFT_TOP - BOARD_SIZE
NUM_LAYERS = 8


def get_model(*args, **kwargs):
  # return get_keras_unet(*args, **kwargs)
  return get_unet_model(*args, **kwargs)


import keras
from keras import layers
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten,
                                     BatchNormalization, Activation, Conv2DTranspose)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
  # first layer
  x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
             padding="same")(input_tensor)
  if batchnorm:
    x = BatchNormalization()(x)
  x = Activation("relu")(x)
  # second layer
  x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
              padding="same")(x)
  if batchnorm:
    x = BatchNormalization()(x)
  x = Activation("relu")(x)
  return x

def get_unet_model(input_shape=(BOARD_SIZE, BOARD_SIZE, 8),
              num_ship_actions=NUM_SHIP_ACTIONS,
              num_shipyard_actions=NUM_SHIPYARD_ACTIONS,
              input_padding=((PADDING_LEFT_TOP, PADDING_RIGHT_BOTTOM),
                             (PADDING_LEFT_TOP, PADDING_RIGHT_BOTTOM)),
                    n_filters=16, dropout=0.5, batchnorm=True):
  inputs = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(input_padding)(inputs)

  # contracting path
  c1 = conv2d_block(x, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
  p1 = MaxPooling2D((2, 2)) (c1)
  p1 = Dropout(dropout*0.5)(p1)

  c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
  p2 = MaxPooling2D((2, 2)) (c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
  p3 = MaxPooling2D((2, 2)) (c3)
  p3 = Dropout(dropout)(p3)

  c6 = p3
  # c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
  # p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
  # p4 = Dropout(dropout)(p4)

  # c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

  # # expansive path
  # u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
  # u6 = concatenate([u6, c4])
  # u6 = Dropout(dropout)(u6)
  # c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

  def decoder():
    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    return c9

  ship_outputs = layers.Conv2D(num_ship_actions, 3,
                                        activation="softmax", padding="same")(decoder())
  ship_outputs = layers.Cropping2D(input_padding, name="ship_crop")(ship_outputs)

  critic_outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(decoder())
  critic_outputs = layers.Cropping2D(input_padding, name="critic_crop")(critic_outputs)

  model = keras.Model(inputs, outputs=[ship_outputs, critic_outputs])
  return model

def get_keras_unet(input_shape=(BOARD_SIZE, BOARD_SIZE, 8),
              num_ship_actions=NUM_SHIP_ACTIONS,
              num_shipyard_actions=NUM_SHIPYARD_ACTIONS,
              input_padding=((PADDING_LEFT_TOP, PADDING_RIGHT_BOTTOM),
                             (PADDING_LEFT_TOP, PADDING_RIGHT_BOTTOM))):
  inputs = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(input_padding)(inputs)

  ### [First half of the network: downsampling inputs] ###
  # Entry block
  x = layers.Conv2D(64, 1, strides=1, padding="same", kernel_initializer='he_normal')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128]:
  # for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer='he_normal')(
      previous_block_activation
    )
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  ### [Second half of the network: upsampling inputs] ###

  def decoder(x):
    previous_block_activation = x  # Set aside next residual
    # for filters in [256, 128, 64, 32]:
    for filters in [128, 64]:
      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer='he_normal')(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer='he_normal')(x)
      x = layers.BatchNormalization()(x)

      x = layers.UpSampling2D(2)(x)

      # Project residual
      residual = layers.UpSampling2D(2)(previous_block_activation)
      residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer='he_normal')(residual)
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual
    return layers.Cropping2D(input_padding)(x)

  encoder_output = x
  ship_outputs = layers.Conv2D(num_ship_actions, 3,
                               activation="softmax", padding="same")(decoder(encoder_output))
  critic_outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(decoder(encoder_output))

  model = keras.Model(inputs, outputs=[ship_outputs, critic_outputs])
  return model

def get_model_small(input_shape=(BOARD_SIZE, BOARD_SIZE, 8),
              num_ship_actions=NUM_SHIP_ACTIONS,
              num_shipyard_actions=NUM_SHIPYARD_ACTIONS,
              input_padding=((PADDING_LEFT_TOP, PADDING_RIGHT_BOTTOM),
                             (PADDING_LEFT_TOP, PADDING_RIGHT_BOTTOM))):
  import keras
  from keras import layers
  from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Flatten
  from keras.regularizers import l2

  inputs = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(input_padding)(inputs)
  x = layers.Conv2D(128, 1, strides=1, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  conv1 = layers.SeparableConv2D(64,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(x)
  conv1 = layers.SeparableConv2D(64,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv1)
  conv1 = layers.BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = layers.SeparableConv2D(128,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool1)
  conv2 = layers.SeparableConv2D(128,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv2)
  conv2 = layers.BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  pool3 = pool2
  # conv3 = layers.SeparableConv2D(256,
                 # 3,
                 # activation='relu',
                 # padding='same',
                 # kernel_initializer='he_normal')(pool2)
  # conv3 = layers.SeparableConv2D(256,
                 # 3,
                 # activation='relu',
                 # padding='same',
                 # kernel_initializer='he_normal')(conv3)
  # conv3 = layers.BatchNormalization()(conv3)
  # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  def decoder(input_tensor):
    # up7 = layers.SeparableConv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    # up7 = layers.SeparableConv2D(256,
                 # 2,
                 # activation='relu',
                 # padding='same',
                 # kernel_initializer='he_normal')(
                     # UpSampling2D(size=(2, 2))(input_tensor))

    # merge7 = concatenate([conv3, up7], axis=3)
    # conv7 = layers.SeparableConv2D(256,
                   # 3,
                   # activation='relu',
                   # padding='same',
                   # kernel_initializer='he_normal')(merge7)
    # conv7 = layers.SeparableConv2D(256,
                   # 3,
                   # activation='relu',
                   # padding='same',
                   # kernel_initializer='he_normal')(conv7)
    # conv7 = layers.BatchNormalization()(conv7)
    conv7 = input_tensor

    up8 = layers.SeparableConv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = layers.SeparableConv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = layers.SeparableConv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = layers.SeparableConv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = layers.SeparableConv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = layers.SeparableConv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    return conv9
    # return layers.Cropping2D(input_padding)(conv9)

  ship_outputs = layers.Conv2D(num_ship_actions, 3,
                                        activation="softmax", padding="same")(decoder(pool3))
  ship_outputs = layers.Cropping2D(input_padding, name="ship_crop")(ship_outputs)

  critic_outputs = layers.Conv2D(1, 3, activation="linear",
                                          padding="same")(decoder(pool3))
  critic_outputs = layers.Cropping2D(input_padding, name="critic_crop")(critic_outputs)

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
            enemy_position_map, enemy_cargo_map, enemy_shipyard_map,
            self.get_border_map()]
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


  def get_border_map(self):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    v[0, :] = v[:, 0] = v[BOARD_SIZE-1, :] = v[:, BOARD_SIZE-1] = 1
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

    if halite_layer:
      v /= HALITE_NORMALIZTION_VAL
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
    self.ship_return_step = {} # ship id => ship born step.

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
    MAX_SHIPYARD_NUM = 5
    MAX_SHIP_AGE = 50

    if not self.me.ship_ids:
      return

    # first ship is not born from shipyard
    # ships = sorted(self.me.ships, key=lambda s: self.ship_return_step.get(s.id, 0))
    ships = self.me.ships
    random.shuffle(ships)
    has_shipyard = len(self.me.shipyard_ids) > 0

    halite = self.me.halite
    for ship in ships:
      # Convert for initial shipyard.
      if (not has_shipyard
          and ship.halite + halite >= (self.c.convert_cost + self.c.spawn_cost)):
        ship.next_action = ShipAction.CONVERT
        halite -= self.c.convert_cost
        break

      # Punish ship not return.
      ship_age = self.board.step - self.ship_return_step[ship.id]
      # print(F'ship[{ship.id}], cargo={ship.halite} age={ship_age}')
      if (has_shipyard and len(self.me.shipyard_ids) < MAX_SHIPYARD_NUM
          and ship.halite + halite >= (self.c.convert_cost + self.c.spawn_cost * 2)
          and ship_age >= MAX_SHIP_AGE):
        ship.next_action = ShipAction.CONVERT
        halite -= self.c.convert_cost
        continue

    self.me._halite = halite

  def spawn_ships(self):
    MAX_SHIP_NUM = 3
    # TODO(wangfei): stop spawn when step > XXX
    if len(self.me.ship_ids) >= MAX_SHIP_NUM:
      return

    halite = self.me.halite
    spawn_count = 0
    for yard in self.me.shipyards:
      if halite >= self.c.spawn_cost:
        yard.next_action = ShipyardAction.SPAWN
        halite -= self.c.spawn_cost
        spawn_count += 1

      if len(self.me.ship_ids) + spawn_count >= MAX_SHIP_NUM:
        break

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

  def update_ship_info(self):
    for ship in self.me.ships:
      if ship.cell.shipyard_id:
        self.ship_return_step[ship.id] = self.board.step

  def execute(self):
    self.update_ship_info()
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
