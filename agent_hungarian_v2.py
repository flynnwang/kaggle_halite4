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
SHIP_TO_SHIYARD_FACTOR = 8

# TODO: estimate this value.
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
      ship.target_cell = None
      ship.next_cell = None

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
    ship.next_cell.is_occupied = ship
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
      neighbor_cells = [
          next_cell, next_cell.north, next_cell.south, next_cell.east,
          next_cell.west
      ]
      for i, nb_cell in enumerate(neighbor_cells):
        if (has_enemy_ship(nb_cell, self.me) and
            nb_cell.ship.halite < ship.halite):
          v += 100
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
    for ship in self.my_idle_ships:
      # if ship.min_enemy_dist > 2 or ship.halite <= MIN_HALITE_BEFORE_HOME:
      if ship.halite <= MIN_HALITE_BEFORE_HOME:
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
    ship = self.board.ships[ship_id]
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
    self.convert_to_shipyard()

    self.continue_mine_halite()
    self.send_ship_to_shipyard()
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
    # Do not spawn ship on a occupied shipyard (or cell).
    if shipyard.cell.ship_id:
      continue

    if me.halite <= MIN_HALITE_TO_BUILD_SHIP:
      continue

    # NOET: do not move ship onto a spawning shipyard.
    me._halite -= board.configuration.spawn_cost
    shipyard.next_action = ShipyardAction.SPAWN
    shipyard.cell.is_occupied = True


def agent(obs, config):
  board = Board(obs, config)

  # Init
  for cell in board.cells.values():
    cell.is_targetd = False
    cell.is_occupied = False

  spawn_ships(board)
  ShipStrategy(board).execute()
  return board.current_player.next_actions
