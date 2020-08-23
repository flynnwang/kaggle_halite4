# -*- coding: utf-8 -*-

import os

import numpy as np
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import evaluate, make

MIN_WEIGHT = -99999

BOARD_SIZE = 21
INPUT_MAP_SIZE = (BOARD_SIZE, BOARD_SIZE)
MODEL_INPUT_SIZE = 15
NUM_LAYERS = 2 + 3*4 + 2  # 12

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

  def __init__(self, board, player_id, prev_board=None):
    self.board = board
    self.me = board.players[player_id]
    self.prev_board = prev_board

  def get_player_input(self, move_axis=False):
    halite_map = self.halite_cell_map()
    cargo_map = self.cargo_map()

    me = self.me
    shipyard_map = self.player_shipyard_map(me.id)
    ship_position_map = self.player_ship_map(me.id, halite_layer=False)

    ship_trace_map = ship_position_map.copy()
    if self.prev_board:
      prev_mi = ModelInput(self.prev_board, me.id)
      ship_trace_map += prev_mi.player_ship_map(me.id, halite_layer=False) * 0.5

    maps = [halite_map,  #0
            cargo_map,   #1
            shipyard_map,#2
            ship_position_map, #4
            ship_trace_map]    #5
    for oppo in self.board.opponents:
      enemy_shipyard_map = self.player_shipyard_map(oppo.id)
      enemy_position_map = self.player_ship_map(oppo.id, halite_layer=False)

      # Enemy's previous positions
      enemy_trace_map = enemy_position_map.copy()
      if self.prev_board:
        prev_mi = ModelInput(self.prev_board, self.me.id)
        enemy_trace_map += prev_mi.player_ship_map(oppo.id, halite_layer=False) * 0.5

      enemy_maps = [enemy_shipyard_map,  #6
                    enemy_position_map,  #7
                    enemy_trace_map,     #8
                    ]
      maps.extend(enemy_maps)

    v = np.stack(maps)
    if move_axis:
      v = np.moveaxis(v, 0, -1)
    return v

  def ship_centered_input(self, ship, player_input, move_axis=True):
    assert ship.player_id == self.me.id

    # assume axis not moved
    CARGO_MAP_INDEX = 1
    ship_cargo = player_input[CARGO_MAP_INDEX,
                              ship.position.x, ship.position.y]

    ENEMY_POS_INDEX = (6, 9, 12)
    dangerous_enemy_map = player_input[ENEMY_POS_INDEX, :, :].sum(axis=0)
    cargo_map = player_input[CARGO_MAP_INDEX, :, :].copy()
    dangerous_enemy_map[cargo_map > ship_cargo] = 0  # ignore enemy with larger halite

    # Ship's current position and it's last position.
    ship_position_map = np.zeros(shape=INPUT_MAP_SIZE)
    ship_position_map[ship.position.x, ship.position.y] = 1
    if self.prev_board:
      prev_ship = self.prev_board.ships.get(ship.id)
      if prev_ship:
        ship_position_map[prev_ship.position.x, prev_ship.position.y] += 0.5

    v = np.stack([dangerous_enemy_map, ship_position_map])
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

  def cargo_map(self):
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player_id in self.board.players:
      v += self.player_ship_map(player_id=player_id, halite_layer=True)
    return v

  def enemy_ship_map(self, halite_layer=True):
    current_player_id = self.me.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.player.values():
      # Skips current player.
      if player.id == current_player_id:
        continue
      v += self.player_ship_map(player_id=player.id, halite_layer=halite_layer)
    return v

  def enemy_shipyard_map(self):
    current_player_id = self.me.id
    v = np.zeros(shape=INPUT_MAP_SIZE)
    for player in self.board.players.values():
      # Skips current player.
      if player.id == current_player_id:
        continue
      v += self.player_shipyard_map(player_id=player.id)
    return v
