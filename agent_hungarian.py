#!/usr/bin/env python

import random
from collections import deque

import numpy as np
from kaggle_environments.envs.halite.helpers import *

# If less than this value, Give up mining more halite from this cell.
MINING_CELL_MIN_HALITE = 30.0

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

# The factor is num_of_ships : num_of_shipyards
SHIP_TO_SHIYARD_FACTOR = 10

MIN_HALITE_BEFORE_HOME = 500


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


def compute_next_moves(source, target):
  if source == target:
    return

  moves = []
  offset = target - source
  sign = np.sign(offset)

  if offset[0] != 0:
    moves.append(Point(sign[0], 0))

  if offset[1] != 0:
    moves.append(Point(0, sign[1]))
  return moves


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


def mining_steps(h, collect_rate):
  # TODO(wangfei): cache f**s
  f = 1.0 - collect_rate
  for s in range(1, 20):
    if h * (f**s) < MINING_CELL_MIN_HALITE:
      break
  return s


# TODO(wangfei): extract a class for this.
def ship_stragegy(board):
  """Sends every ships to the nearest cell with halite."""
  me = board.current_player

  halite_cells = []
  for cell in board.cells.values():
    cell.has_mining_plan = False
    cell.has_ally_ship = False
    if cell.halite > MINING_CELL_MIN_HALITE:
      halite_cells.append(cell)

  # Init ship properties.
  ships = me.ships
  for ship in ships:
    ship.is_stay = False

  # TODO(wangfei): may need to shift
  # Ship that stay on current haltie.
  # for ship in ships:
  # if ship.cell.halite > MINING_CELL_MIN_HALITE:
  # ship.next_action = None
  # ship.is_stay = True
  # ship.cell.has_ally_ship = True

  def ship_stay(ship):
    ship.next_action = None
    ship.is_stay = True
    ship.cell.has_ally_ship = True

  def try_move(ship, target_cell):
    """Move ship towards the target cell without collide with allies."""
    moves = compute_next_moves(ship.position, target_cell.position)
    if not moves:
      return False

    for move in moves:
      next_cell = ship.cell.neighbor(move)
      if next_cell.has_ally_ship:
        continue

      ship.next_action = direction_to_ship_action(move)
      target_cell.has_mining_plan = True
      next_cell.has_ally_ship = True
      return True
    return False

  # Ship goes back home.
  for ship in ships:
    if ship.next_action or ship.is_stay or ship.halite <= MIN_HALITE_BEFORE_HOME:
      continue

    min_dist = 99999
    min_dist_yard = None
    for y in me.shipyards:
      # Skip go-back to yard that going to spawn and dist == 1.
      d = manhattan_dist(ship.position, y.position, board.configuration.size)
      if y.next_action == ShipyardAction.SPAWN and d == 1:
        continue

      if d < min_dist:
        min_dist = d
        min_dist_yard = y

    if min_dist_yard and try_move(ship, min_dist_yard.cell):
      continue
    else:
      ship_stay(ship)

  # Ship that goes to halite.
  for ship in ships:
    if ship.next_action or ship.is_stay:
      continue

    max_expected_return = 0
    max_cell = None
    for c in halite_cells:
      if c.has_mining_plan:
        continue

      # TODO(wangfei): use search.
      # Manhattan move is short-sighted, since it will not get round blocking
      # cells to move.
      stay_steps = mining_steps(c.halite, board.configuration.collect_rate)
      move_steps = manhattan_dist(ship.position, c.position,
                                  board.configuration.size)
      total_steps = stay_steps + move_steps
      expect_return = c.halite / total_steps
      if expect_return > max_expected_return:
        max_expected_return = expect_return
        max_cell = c

    if max_cell and try_move(ship, max_cell):
      continue
    else:
      ship_stay(ship)


def build_shipyard(board):
  """Builds shipyard with a random ship if we have enough halite and ships."""

  # TODO: select a far-away ship to convert?
  me = board.current_player
  if me.halite <= MIN_HALITE_TO_BUILD_SHIPYARD or not me.ship_ids:
    return

  # Keep balance for the number of ships and shipyards.
  num_ships = len(me.ship_ids)
  num_shipyards = len(me.shipyard_ids)
  if num_shipyards * SHIP_TO_SHIYARD_FACTOR >= num_ships:
    return

  # Only build one shipyard at a time.
  me._halite -= board.configuration.convert_cost
  ship_id = random.sample(me.ship_ids, k=1)[0]
  ship = board.ships[ship_id]
  ship.next_action = ShipAction.CONVERT


def spawn_ships(board):
  """Spawns ships if we have enough money and no collision with my own ships."""
  me = board.current_player

  shipyards = me.shipyards
  random.shuffle(shipyards)

  for shipyard in shipyards:
    # Do not spawn ship on a occupied shipyard (or cell).
    if shipyard.cell.ship_id:
      continue

    if me.halite <= MIN_HALITE_TO_BUILD_SHIP:
      continue

    # NOET: do not move ship onto a spawning shipyard.
    me._halite -= board.configuration.spawn_cost
    shipyard.next_action = ShipyardAction.SPAWN


def agent(obs, config):
  board = Board(obs, config)

  spawn_ships(board)

  build_shipyard(board)

  ship_stragegy(board)

  return board.current_player.next_actions
