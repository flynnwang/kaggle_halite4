# -*- coding: utf-8 -*-


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


OFFSET = Point(5, 5)

BOARD_SIZE = 21
INPUT_MAP_SIZE = (32, 32)
HALITE_NORMALIZTION_VAL = 500.0
TOTAL_STEPS = 400


class Replayer:

  def __init__(self, strategy, replay_json, player_id=0):
    self.strategy = strategy
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = make("halite", configuration=replay_json['configuration'],
                    steps=replay_json['steps'])
    self.step = 0

  def get_board(self, step):
    state = self.replay_json['steps'][step][0]
    obs = state['observation']
    obs['player'] = self.player_id
    return Board(obs, self.env.configuration)

  def simulate(self, step=0):
    board = self.get_board(step)
    self.strategy.update(board)
    self.strategy.execute()
    self.step += 1

  def play(self, steps=1):
    for i in range(steps):
      self.simulate(i)


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

  def get_input(self):
    halites = self.halite_cell_map()

    me = self.board.current_player
    my_ship_map = self.player_ship_map(player_id=me.id)
    my_shipyard_map = self.player_shipyard_map(player_id=me.id)

    enemy_ship_map = self.enemy_ship_map()
    enemy_shipyard_map = self.enemy_shipyard_map()

    aux_map = self.get_auxiliary_map()

    maps = [halites, my_ship_map, my_shipyard_map, enemy_ship_map,
            enemy_shipyard_map, aux_map]
    return [shift(m, OFFSET) for m in maps]

  def get_auxiliary_map(self):
    v = np.zeros(shape=INPUT_MAP_SIZE)

    # progress
    v[0, 0] = (self.board.step + 1) / TOTAL_STEPS

    total_ships = sum(len(p.ship_ids) for p in self.board.players.values()) or 1
    total_halite = sum(p.halite for p in self.board.players.values()) or 1
    total_cargo = sum(cargo(p) for p in self.board.players.values()) or 1

    # my ships
    me = self.board.current_player
    v[5, 0] = len(me.ship_ids) / 50
    v[5, 5] = len(me.ship_ids) / total_ships

    # my halite
    v[10, 0] = me.halite / HALITE_NORMALIZTION_VAL
    v[10, 5] = me.halite / total_halite

    # my cargo
    v[15, 0] = cargo(me) / HALITE_NORMALIZTION_VAL
    v[15, 5] = cargo(me) / total_cargo
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
