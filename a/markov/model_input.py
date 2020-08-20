# -*- coding: utf-8 -*-

import os

import numpy as np
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import evaluate, make

MIN_WEIGHT = -99999

BOARD_SIZE = 21
INPUT_MAP_SIZE = (BOARD_SIZE, BOARD_SIZE)
MODEL_INPUT_SIZE = 15
NUM_LAYERS = 9

HALITE_NORMALIZTION_VAL = 1000.0
MAX_SHIP_CARGO = 2000.0

SHIP_ACTIONS = [a for a in list(ShipAction) if a != ShipAction.CONVERT] + [None]
NUM_SHIP_ACTIONS = len(SHIP_ACTIONS)


def cargo(player):
  """Computes the cargo value for a player."""
  return sum([s.halite for s in player.ships], 0)


class ModelInput:
  """Convert a board state into 7x7xX model input for a ship of current player.
  """

  def __init__(self, board, player_id):
    self.board = board
    self.me = board.players[player_id]

  def get_player_input(self, move_axis=False):
    halites = self.halite_cell_map()
    me = self.me
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

  def ship_centered_input(self, ship, player_input, move_axis=True):
    assert ship.player_id == self.me.id

    # assume axis not moved
    ship_cargo = player_input[2, ship.position.x, ship.position.y]

    enemy_cargo_map = player_input[5, :, :]
    danger_enemy_map = player_input[4, :, :].copy() # enemy position
    danger_enemy_map[enemy_cargo_map > ship_cargo] = 0

    ship_position_map = np.zeros(shape=INPUT_MAP_SIZE)
    ship_position_map[ship.position.x, ship.position.y] = 1

    v = np.stack([danger_enemy_map, ship_position_map])
    v = np.concatenate([player_input, v], axis=0)

    v = np.roll(v, shift=(MODEL_INPUT_SIZE//2 - ship.position.x), axis=1)
    v = np.roll(v, shift=(MODEL_INPUT_SIZE//2 - ship.position.y), axis=2)
    v = v[:, :MODEL_INPUT_SIZE, :MODEL_INPUT_SIZE]

    if move_axis:
      v = np.moveaxis(v, 0, -1)
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
      value = min(ship.halite, MAX_SHIP_CARGO) if halite_layer else 1
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
    current_player_id = self.me.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.players.values():
      if player_id != None and player.id != player_id:
        continue
      v += self.player_ship_map(player_id=player.id, halite_layer=halite_layer)
    return v

  def enemy_shipyard_map(self, player_id=None):
    current_player_id = self.me.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.players.values():
      if player_id != None and player.id != player_id:
        continue
      v += self.player_shipyard_map(player_id=player.id)
    return v
