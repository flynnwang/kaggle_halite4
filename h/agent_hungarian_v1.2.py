#!/usr/bin/env python
"""Tests expect_return with halite growth.

ACCEPTED.

Total Matches: 141 | Matches Queued: 9
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
Hungarian v1.2                 | bHoZ3UZNq0SD   | 27.0110296      | μ=29.618, σ=0.869  | 74
Hungarian v1                   | m0usb391ug9H   | 25.7703241      | μ=28.361, σ=0.864  | 74
Hungarian v1.1                 | tmo024Tnqvva   | 25.7510054      | μ=28.345, σ=0.865  | 67
swarm                          | VEdhOravyNCX   | 24.8444348      | μ=27.318, σ=0.825  | 78
Manhattan                      | kZXLGgfVKGXh   | 16.5500997      | μ=18.983, σ=0.811  | 86
somebot                        | FErt8IPHKdWh   | 14.4427869      | μ=16.893, σ=0.817  | 91
stillbot                       | M2y9hkW0ChhZ   | 11.1102277      | μ=13.773, σ=0.888  | 94
"""

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
SHIP_TO_SHIYARD_FACTOR = 7

# TODO: estimate this value.
MIN_HALITE_BEFORE_HOME = 300

MAX_SHIP_NUM = 23


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

  if direction == Point(0, 0):
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
  """Sends every ships to the nearest cell with halite.


  cell:
    |has_mining_plan|: prevent more than 1 alley ships choose the same halite.
    |has_ally_ship|: will has alley ship in the next round.

  ship:
    |next_cell: next cell location of the ship.
    |is_stay|: will stay on the current cell.
  """
  me = board.current_player

  halite_cells = []
  for cell in board.cells.values():
    if cell.halite > MINING_CELL_MIN_HALITE:
      halite_cells.append(cell)

  # Init ship properties.
  ships = me.ships
  opponents = board.opponents
  for ship in ships:
    ship.is_stay = False
    ship.next_cell = ship.cell

    # Compute min enemy distance

    ship.min_enemy_dist = 999
    if opponents:
      enemy_dists = [
          manhattan_dist(ship.position, enemy_ship.position,
                         board.configuration.size)
          for e in opponents
          for enemy_ship in e.ships
          # TODO: this is not accurate
          if enemy_ship.halite < ship.halite
      ]
      if enemy_dists:
        ship.min_enemy_dist = min(enemy_dists)

  # TODO(wangfei): may need to shift
  # Ship that stay on current haltie.
  # for ship in ships:
  # if ship.cell.halite > MINING_CELL_MIN_HALITE:
  # ship.next_action = None
  # ship.is_stay = True
  # ship.cell.has_ally_ship = True

  def ship_stay(ship):
    ship.next_action = None
    ship.cell.has_ally_ship = ship
    ship.is_stay = True
    ship.next_cell = ship.cell

  def try_move(ship, target_cell):
    """Move ship towards the target cell without collide with allies."""
    target_cell.has_mining_plan = True

    if ship.position == target_cell.position:
      ship_stay(ship)
      return True

    moves = compute_next_moves(ship.position, target_cell.position)
    if not moves:
      return False

    for move in moves:
      next_cell = ship.cell.neighbor(move)
      if next_cell.has_ally_ship:
        continue

      ship.next_action = direction_to_ship_action(move)
      ship.next_cell = next_cell
      next_cell.has_ally_ship = ship
      return True
    return False

  def max_expected_return_cell(ship):
    growth = board.configuration.regen_rate + 1.0

    max_cell = None
    max_expected_return = 0
    for c in halite_cells:
      if c.has_mining_plan and c.has_ally_ship:
        continue

      move_steps = manhattan_dist(ship.position, c.position,
                                  board.configuration.size)
      total_steps = move_steps + 1
      expected_halite = min(c.halite * (growth**move_steps), 500)
      expect_return = expected_halite / total_steps
      if expect_return > max_expected_return:
        max_expected_return = expect_return
        max_cell = c
    return max_cell, max_expected_return

  # Ship that stay on halite cell.
  # NOTE: this way make a ship stay on the way home.
  for ship in ships:
    if ship.next_action or ship.is_stay:
      continue

    max_cell, _ = max_expected_return_cell(ship)
    if max_cell and max_cell.position == ship.position:
      max_cell.has_mining_plan = True
      ship_stay(ship)

  # Ship goes back home.
  for ship in ships:
    if ship.next_action or ship.is_stay:
      continue

    # if ship.min_enemy_dist > 2 or ship.halite <= MIN_HALITE_BEFORE_HOME:
    if ship.halite <= MIN_HALITE_BEFORE_HOME:
      continue

    min_dist = 99999
    min_dist_yard = None
    for y in me.shipyards:
      d = manhattan_dist(ship.position, y.position, board.configuration.size)
      if d < min_dist:
        min_dist = d
        min_dist_yard = y

    if min_dist_yard and try_move(ship, min_dist_yard.cell):
      continue
    ship_stay(ship)

  # Ship that goes to halite.
  for ship in ships:
    if ship.next_action or ship.is_stay:
      continue

    max_cell, _ = max_expected_return_cell(ship)
    if max_cell and try_move(ship, max_cell):
      continue
    ship_stay(ship)

  # Collision avoid.
  while True:
    found = False
    cell_ship_count = {}
    for ship in ships:
      c = ship.next_cell
      cell_ship_count[c] = cell_ship_count.get(c, 0) + 1

    for ship in ships:
      if ship.is_stay or not ship.next_action:
        continue
      if cell_ship_count[ship.next_cell] > 1:
        # print('Step ', board.step, 'Collision avoid to ship at ', ship.position)
        ship_stay(ship)
        found = True

    if not found:
      break


def build_shipyard(board):
  """Builds shipyard with a random ship if we have enough halite and ships."""
  convert_cost = board.configuration.convert_cost

  # TODO: select a far-away ship to convert?
  me = board.current_player
  if not me.ship_ids or me.halite < convert_cost:
    return

  if me.shipyards and me.halite < MIN_HALITE_TO_BUILD_SHIPYARD:
    return

  # Keep balance for the number of ships and shipyards.
  num_ships = len(me.ship_ids)
  num_shipyards = len(me.shipyard_ids)
  if num_shipyards * SHIP_TO_SHIYARD_FACTOR >= num_ships:
    return

  # Only build one shipyard at a time.
  me._halite -= convert_cost
  ship_id = random.sample(me.ship_ids, k=1)[0]
  ship = board.ships[ship_id]
  ship.next_action = ShipAction.CONVERT


def spawn_ships(board):
  """Spawns ships if we have enough money and no collision with my own ships."""
  me = board.current_player

  num_ships = len(me.ship_ids)
  if num_ships >= MAX_SHIP_NUM:
    return

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
    shipyard.cell.has_ally_ship = True


def agent(obs, config):
  board = Board(obs, config)

  # Init
  for cell in board.cells.values():
    cell.has_mining_plan = False
    cell.has_ally_ship = None

  spawn_ships(board)

  build_shipyard(board)

  ship_stragegy(board)

  return board.current_player.next_actions
