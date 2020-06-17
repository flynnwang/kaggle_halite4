#!/usr/bin/env python

from collections import deque

import numpy as np
from kaggle_environments.envs.halite.helpers import *

MINING_CELL_MIN_HALITE = 10.0


def manhattan_dist(a, b, size):

  def dist(x, y):
    v = abs(x - y)
    return min(v, size - v)

  return dist(a.x, b.x) + dist(a.y, b.y)


def compute_next_move(source, target):
  if source == target:
    return

  offset = target - source
  idx = np.argmax(abs(offset))
  sign = np.sign(offset)[idx]
  direction = Point(sign, 0) if idx == 0 else Point(0, sign)
  return direction


def direction_to_ship_action(direction):
  if direction is None:
    return None

  if direction == Point(0, 1):
    return ShipAction.NORTH
  if direction == Point(1, 0):
    return ShipAction.EAST
  if direction == Point(0, -1):
    return ShipAction.SOUTH
  if direction == Point(-1, 0):
    return ShipAction.WEST
  assert False


def mine_halite_plan(board):
  """Sends every ships to the nearest cell with halite."""
  me = board.current_player

  halite_cells = []
  for cell in board.cells.values():
    if cell.halite > MINING_CELL_MIN_HALITE:
      halite_cells.append(cell)
      # cell.has_mining_plan = None

  for ship in me.ships:
    min_dist = 99999
    min_dist_cell = None
    for cell in halite_cells:
      # if cell.has_mining_plan:
      # continue

      d = manhattan_dist(ship.position, cell.position, board.configuration.size)
      if d < min_dist:
        min_dist = d
        min_dist_cell = cell

    if min_dist_cell:
      direction = compute_next_move(ship.position, min_dist_cell.position)
      ship.next_action = direction_to_ship_action(direction)


def agent(obs, config):
  board = Board(obs, config)

  mine_halite_plan(board)

  # Set actions for each shipyard
  me = board.current_player
  for shipyard in me.shipyards:
    shipyard.next_action = None
  return me.next_actions
