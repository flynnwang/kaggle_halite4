#!/usr/bin/env python
"""
Tests use mean halite value in cell as a trigger for return home.

ACCEPTED.

Tournament - ID: mN1WFt, Name: Your Halite 4 Trueskill Ladder | Dimension - ID: pipIt6, Name: Halite 4 Dimension
Status: running | Competitors: 9 | Rank System: trueskill

Total Matches: 3005 | Matches Queued: 61
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
hungarian v2.2                 | b4iPu5HJQdyJ   | 37.0251603      | μ=39.846, σ=0.940  | 1102
hungarian v2.1                 | gmpjAUyAf8wq   | 31.7964190      | μ=34.227, σ=0.810  | 1103
hungarian v2                   | ZE9zoBgw8ItD   | 31.3663609      | μ=33.736, σ=0.790  | 1058
swarm                          | q2oMzXtuszW3   | 23.3463794      | μ=25.459, σ=0.704  | 1327
hungarian v1                   | TuQctmopnt3V   | 21.1654487      | μ=23.327, σ=0.721  | 1473
hungarian v1.2                 | tfMuYqIYnD5k   | 20.2444771      | μ=22.394, σ=0.716  | 1469
manhattan                      | sgD3YXRz2kfU   | 17.6538227      | μ=19.763, σ=0.703  | 1444
somebot                        | FYxDlvKmP6Lh   | 15.7972051      | μ=17.944, σ=0.716  | 1501
stillbot-1                     | XdIuEDzZP6FN   | 14.0172679      | μ=16.254, σ=0.746  | 1495
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
SHIP_TO_SHIYARD_FACTOR = 8

MIN_HALITE_BEFORE_HOME = 300

MAX_SHIP_NUM = 17


def manhattan_dist(a: Point, b: Point, size):

  def dist(x, y):
    v = abs(x - y)
    return min(v, size - v)

  return dist(a.x, b.x) + dist(a.y, b.y)


def has_enemy_ship(cell, me):
  if not cell.ship_id:
    return False

  ship = cell.ship
  return ship.player_id != me.id


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


def get_neighbor_cells(cell, include_self=False):
  neighbor_cells = [cell] if include_self else []
  neighbor_cells.extend([cell.north, cell.south, cell.east, cell.west])
  return neighbor_cells


P_STAY_ON_HALITE = 1000
P_MOVE_TO_HALITE = 900
P_RETURN_TO_YARD = 800


class ShipStrategy:
  """Sends every ships to the nearest cell with halite.


  cell:
    |is_targetd|: prevent more than 1 alley ships choose the same halite.
    |is_occupied|: will has ship in the next round.

  ship:
    |next_cell: next cell location of the ship.
    |has_assignment|: has already has task assignment for this ship.
    |target_cell|: send ship to this cell, may be the same of current cell, None by default.
    |priority|: used to rank ship for moves.
  """

  def __init__(self, board):
    self.board = board
    self.me = board.current_player

    # Init halite cells
    self.halite_cells = []
    for cell in board.cells.values():
      if cell.halite > MINING_CELL_MIN_HALITE:
        self.halite_cells.append(cell)

    # Default ship to stay on the same cell without assignment.
    ships = self.me.ships
    for ship in ships:
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.priority = 0

    # Statistics values
    self.mean_halite_value = MIN_HALITE_BEFORE_HOME

  def data_analyis(self):
    if self.halite_cells:
      halite_values = [c.halite for c in self.halite_cells]
      self.mean_halite_value = np.mean(halite_values)

  @property
  def step(self):
    return self.board.step

  @property
  def my_idle_ships(self):
    for ship in self.me.ships:
      if ship.next_action or ship.has_assignment:
        continue
      yield ship

  @staticmethod
  def ship_move_task(ship, target_cell: Cell, priority=0):
    ship.has_assignment = True
    ship.target_cell = target_cell
    ship.target_cell.is_targetd = True
    ship.priority = priority

  @staticmethod
  def ship_stay(ship):
    ship.next_action = None
    ship.next_cell = ship.cell
    ship.next_cell.is_occupied = True
    if ship.cell.halite > 0:
      ship.cell.is_targetd = True

  def rank_next_moves(self, ship: Ship, target):
    """Smaller values is better"""
    source = ship.position

    board = self.board
    board_size = board.configuration.size
    moves = [Point(0, 0), Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)]

    def rank_func(m):
      next_position = source + m
      v = manhattan_dist(next_position, target, board_size)

      # If there is an enemy in next_position or nearby with lower halite
      next_cell = board[next_position]
      if has_enemy_ship(next_cell,
                        self.me) and next_cell.ship.halite < ship.halite:
        v += 1000

      for i, nb_cell in enumerate(get_neighbor_cells(next_cell)):
        if has_enemy_ship(nb_cell, self.me):
          if nb_cell.ship.halite < ship.halite:
            v += 1000
          # if nb_cell.ship.halite == ship.halite:
          # v += 20
      return v

    moves.sort(key=rank_func)
    return moves

  def take_move(self, ship):
    """Move ship towards the target cell without collide with allies.
    NOTE: can move far away to make room to other ship."""
    moves = self.rank_next_moves(ship, ship.target_cell.position)
    # print('ranked_moves: ', moves)
    if not moves:
      return False

    for move in moves:
      next_cell = ship.cell.neighbor(move)
      if next_cell.is_occupied:
        continue

      ship.next_action = direction_to_ship_action(move)
      ship.next_cell = next_cell
      next_cell.is_occupied = True
      return True
    return False

  def max_expected_return_cell(self, ship):
    growth = self.board.configuration.regen_rate + 1.0

    max_cell = None
    max_expected_return = 0
    for c in self.halite_cells:
      if c.is_targetd or has_enemy_ship(c, self.me):
        continue

      move_steps = manhattan_dist(ship.position, c.position,
                                  self.board.configuration.size)
      expected_halite = min(c.halite * (growth**move_steps),
                            self.board.configuration.max_cell_halite)
      expect_return = expected_halite / (move_steps + 1)
      if expect_return > max_expected_return:
        max_cell = c
        max_expected_return = expect_return
    return max_expected_return, max_cell

  def find_nearest_shipyard(self, ship):
    min_dist = 99999
    min_dist_yard = None
    for y in self.me.shipyards:
      d = manhattan_dist(ship.position, y.position,
                         self.board.configuration.size)
      if d < min_dist:
        min_dist = d
        min_dist_yard = y
    return min_dist, min_dist_yard

  def continue_mine_halite(self):
    """ Ship that stay on halite cell."""
    for ship in self.my_idle_ships:
      # if ship.cell.halite > MINING_CELL_MIN_HALITE:

      _, max_cell = self.max_expected_return_cell(ship)
      if max_cell and max_cell.position == ship.position:
        self.ship_move_task(ship, max_cell, P_STAY_ON_HALITE)

  def send_ship_to_shipyard(self):
    """Ship goes back home after collected enough halite."""
    threshold = int(max(self.mean_halite_value * 3, 100))
    for ship in self.my_idle_ships:
      # if ship.halite <= MIN_HALITE_BEFORE_HOME:
      if ship.halite < threshold:
        continue

      # TODO: if too many ships are home, shall we wait?
      min_dist, min_dist_yard = self.find_nearest_shipyard(ship)
      if min_dist_yard:
        self.ship_move_task(ship, min_dist_yard.cell, P_RETURN_TO_YARD)

  def send_ship_to_halite(self):
    """Ship that goes to halite."""
    for ship in self.my_idle_ships:
      _, max_cell = self.max_expected_return_cell(ship)
      if max_cell:
        self.ship_move_task(ship, max_cell, P_MOVE_TO_HALITE)

  def collision_avoid(self):
    ships = self.me.ships
    while True:
      found = False
      cell_ship_count = {}
      for ship in ships:
        c = ship.next_cell
        cell_ship_count[c] = cell_ship_count.get(c, 0) + 1

      for ship in ships:
        if ship.next_cell == ship.cell:
          continue
        if cell_ship_count[ship.next_cell] > 1:
          self.ship_stay(ship)
          found = True

      if not found:
        break

  def convert_to_shipyard(self):
    """Builds shipyard with a random ship if we have enough halite and ships."""
    convert_cost = self.board.configuration.convert_cost

    # TODO: select a far-away ship to convert?
    me = self.me

    # No ship or money.
    if not me.ship_ids or me.halite < convert_cost:
      return

    # TODO: use function for this threshold
    threshold = MIN_HALITE_TO_BUILD_SHIPYARD
    if self.board.step <= 40:
      threshold = convert_cost
    if me.shipyards and me.halite < threshold:
      return

    # Keep balance for the number of ships and shipyards.
    num_ships = len(me.ship_ids)
    num_shipyards = len(me.shipyard_ids)
    if num_shipyards * SHIP_TO_SHIYARD_FACTOR >= num_ships:
      return

    valid_ships = [s for s in me.ships if s.cell.shipyard_id is None]
    if not valid_ships:
      return

    # Only build one shipyard at a time.
    me._halite -= convert_cost

    ship = random.sample(valid_ships, k=1)[0]
    ship.next_action = ShipAction.CONVERT
    ship.has_assignment = True
    ship.cell.is_targetd = True
    ship.cell.is_occupied = True

  def compute_ship_moves(self):
    """Computes ship moves to maximize the total score."""
    # Skip only convert ships.
    ships = [s for s in self.me.ships if not s.next_action]
    ships.sort(key=lambda s: s.priority, reverse=True)

    # print("move candidates: ", [s.id for s in ships])
    for ship in ships:
      self.take_move(ship)

  def execute(self):
    self.continue_mine_halite()
    self.send_ship_to_shipyard()

    # TODO: add priority for leaving if a ship is converted on yard.
    self.send_ship_to_halite()
    self.compute_ship_moves()

    # TODO: maybe no longer need it?
    self.collision_avoid()


def spawn_ships(board):
  """Spawns ships if we have enough money and no collision with my own ships."""
  me = board.current_player

  num_ships = len(me.ship_ids)
  if num_ships >= MAX_SHIP_NUM:
    return

  shipyards = me.shipyards
  random.shuffle(shipyards)

  for shipyard in shipyards:
    # Skip if not enough money.
    build_ship_threshold = MIN_HALITE_TO_BUILD_SHIP
    if board.step <= 40:
      build_ship_threshold = board.configuration.spawn_cost
    if num_ships and me.halite < build_ship_threshold:
      continue

    # If there is a ship on shipyard and no free neighbor cells.
    yard_cell = shipyard.cell
    if yard_cell.ship_id:
      num_free_cells = sum(
          1 for c in get_neighbor_cells(yard_cell) if c.ship_id is None)
      if num_free_cells == 0:
        continue

    # NOET: do not move ship onto a spawning shipyard.
    me._halite -= board.configuration.spawn_cost
    shipyard.next_action = ShipyardAction.SPAWN
    yard_cell.is_occupied = True

    # No more ships if it's enough.
    num_ships += 1
    if num_ships >= MAX_SHIP_NUM:
      break


def agent(obs, config):
  board = Board(obs, config)

  # Init
  for cell in board.cells.values():
    cell.is_targetd = False
    cell.is_occupied = False

  strategy = ShipStrategy(board)
  strategy.data_analyis()

  strategy.convert_to_shipyard()

  spawn_ships(board)

  strategy.execute()
  return board.current_player.next_actions
