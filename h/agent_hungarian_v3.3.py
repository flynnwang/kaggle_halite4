#!/usr/bin/env python
"""
Plans to remove min halite threshold.

ACCEPTED.

-=-=-=-=-=-=-=-=-=-=-=-|  Your Halite 4 Trueskill Ladder |-=-=-=-=-=-=-=-=-=-=-=-

Tournament - ID: RVGcMV, Name: Your Halite 4 Trueskill Ladder | Dimension - ID: M4twhG, Name: Halite 4 Dimension
Status: running | Competitors: 11 | Rank System: trueskill

Total Matches: 706 | Matches Queued: 67
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
v3.3 no min halite             | LZxXUALbXWo9   | 31.2623540      | μ=33.556, σ=0.765  | 216
v3 ghost                       | wprnyR65WNnQ   | 30.5124029      | μ=32.776, σ=0.755  | 208
v3.2 yield                     | vNDF3UvrCMMb   | 30.1198406      | μ=32.389, σ=0.756  | 196
v2.2.1                         | v6BLsoA9mrIO   | 27.7501136      | μ=29.922, σ=0.724  | 256
v2.1                           | lNQ4FWt2d1ry   | 25.6874985      | μ=27.828, σ=0.714  | 271
swarm                          | jOvx5JekCwGS   | 25.4489304      | μ=27.530, σ=0.694  | 265
v1.2                           | umVlsgXrXx3g   | 19.7268570      | μ=21.848, σ=0.707  | 264
v1                             | 3ohbyoZbnmpK   | 18.8615405      | μ=21.017, σ=0.718  | 262
manhattan                      | rb7eZhVBDEe5   | 15.4508144      | μ=17.637, σ=0.729  | 294
somebot                        | 7tLKHAYMAfzH   | 14.7076806      | μ=16.949, σ=0.747  | 265
stillbot-1                     | 9SIwWh6k4I5h   | 14.1004570      | μ=16.327, σ=0.742  | 283

"""

import random
from enum import Enum, auto
from collections import deque

import numpy as np
from kaggle_environments.envs.halite.helpers import *

# If less than this value, Give up mining more halite from this cell.
MINING_CELL_MIN_HALITE = 0.0

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

# The factor is num_of_ships : num_of_shipyards
SHIP_TO_SHIYARD_FACTOR = 8

MIN_HALITE_BEFORE_HOME = 100
MIN_HALITE_FACTOR = 3

MAX_FARMER_SHIP_NUM = 13
MAX_SHIP_NUM = 20


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


def get_neighbor_cells(cell, include_self=False):
  neighbor_cells = [cell] if include_self else []
  neighbor_cells.extend([cell.north, cell.south, cell.east, cell.west])
  return neighbor_cells


class ShipType(Enum):
  UNKNOWN_TYPE = auto()

  # Ship that collect halite.
  FARMER_TYPE = auto()

  # Ship that bomb enemy's shipyard.
  GHOST_TYPE = auto()


P_STAY_ON_HALITE = 1000
P_MOVE_TO_HALITE = 900
P_RETURN_TO_YARD = 800
P_DESTORY_ENEMY_YARD = 500


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

  SHIP_ID_TO_TYPE = {}

  def __init__(self, board):
    self.board = board
    self.me = board.current_player

    # Init halite cells
    self.halite_cells = []
    for cell in board.cells.values():
      if cell.halite > MINING_CELL_MIN_HALITE:
        self.halite_cells.append(cell)

    # Default ship to stay on the same cell without assignment.
    # print('#', self.step)
    ships = self.me.ships
    has_enough_farmer = len(ships) > MAX_FARMER_SHIP_NUM

    for ship in ships:
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.priority = 0

      # Assign ship type to new ship only.
      if ship.id not in self.SHIP_ID_TO_TYPE:
        self.SHIP_ID_TO_TYPE[ship.id] = ShipType.FARMER_TYPE
        if has_enough_farmer and ship.halite == 0:
          self.SHIP_ID_TO_TYPE[ship.id] = ShipType.GHOST_TYPE
          # print('ghost ship: ', ship.id)

  def collect_game_info(self):
    self.mean_halite_value = MIN_HALITE_BEFORE_HOME
    if self.halite_cells:
      halite_values = [c.halite for c in self.halite_cells]
      self.mean_halite_value = np.mean(halite_values)

    self.max_halite = self.me.halite
    self.max_halite_player_id = self.me.id
    for p in self.board.opponents:
      if p.halite >= self.max_halite:
        self.max_halite = p.halite
        self.max_halite_player_id = p.id
    # print(self.max_halite, self.max_halite_player_id,
    # self.max_halite_player_id == self.me.id)

  @property
  def step(self):
    return self.board.step

  @property
  def my_idle_farmer_ships(self):
    """All ships of type FARMER_TYPE and has no has_assignment."""
    for ship in self.me.ships:
      if self.SHIP_ID_TO_TYPE[ship.id] != ShipType.FARMER_TYPE:
        continue

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

  def rank_next_moves(self, ship: Ship, next_move_positions):
    """Smaller values is better"""
    source = ship.position
    target = ship.target_cell.position
    # TODO: consider friend ship.

    board = self.board
    board_size = board.configuration.size
    moves = [Point(0, 0), Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)]

    is_ghost_ship = self.SHIP_ID_TO_TYPE[ship.id] == ShipType.GHOST_TYPE

    def rank_func(m):
      next_position = source + m
      v = manhattan_dist(next_position, target, board_size)

      # If |next_position| has already taken by my other ships.
      if next_position in next_move_positions:
        v += 50

      # If there is an enemy in next_position or nearby with lower halite
      next_cell = board[next_position]
      if has_enemy_ship(next_cell,
                        self.me) and next_cell.ship.halite < ship.halite:
        v += 1000

      for i, nb_cell in enumerate(get_neighbor_cells(next_cell)):
        if has_enemy_ship(nb_cell, self.me):
          if nb_cell.ship.halite < ship.halite:
            v += 1000

          if is_ghost_ship and nb_cell.ship.halite == ship.halite:
            v += 50
      return v

    moves.sort(key=rank_func)
    return moves

  def take_move(self, ship, next_move_positions):
    """Move ship towards the target cell without collide with allies.
    NOTE: can move far away to make room to other ship."""
    moves = self.rank_next_moves(ship, next_move_positions)
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

  def find_nearest_shipyard(self, ship, shipyards):
    min_dist = 99999
    min_dist_yard = None
    for y in shipyards:
      d = manhattan_dist(ship.position, y.position,
                         self.board.configuration.size)
      if d < min_dist:
        min_dist = d
        min_dist_yard = y
    return min_dist, min_dist_yard

  def continue_mine_halite(self):
    """ Ship that stay on halite cell."""
    for ship in self.my_idle_farmer_ships:
      _, max_cell = self.max_expected_return_cell(ship)
      if max_cell and max_cell.position == ship.position:
        self.ship_move_task(ship, max_cell, P_STAY_ON_HALITE)

  def send_ship_to_shipyard(self):
    """Ship goes back home after collected enough halite."""
    threshold = int(
        max(self.mean_halite_value * MIN_HALITE_FACTOR, MIN_HALITE_BEFORE_HOME))
    for ship in self.my_idle_farmer_ships:
      # if ship.halite <= MIN_HALITE_BEFORE_HOME:
      if ship.halite < threshold:
        continue

      # TODO: if too many ships are home, shall we wait?
      _, min_dist_yard = self.find_nearest_shipyard(ship, self.me.shipyards)
      if min_dist_yard:
        self.ship_move_task(ship, min_dist_yard.cell, P_RETURN_TO_YARD)

  @property
  def my_ghost_ships(self):
    for ship_id in self.me.ship_ids:
      if self.SHIP_ID_TO_TYPE[ship_id] == ShipType.GHOST_TYPE:
        yield self.board.ships[ship_id]

  @property
  def enemy_shipyards(self):
    for e in self.board.opponents:
      for y in e.shipyards:
        yield y

  def send_ship_to_enemy_shipyard(self):
    """Having enough farmers, let's send ghost to enemy shipyard."""

    def non_targeted_enemy_shipyards():
      for y in self.enemy_shipyards:
        if not y.cell.is_targetd:
          yield y

    # do_print = False
    # if len(list(self.my_ghost_ships)) > len(
    # list(non_targeted_enemy_shipyards())):
    # do_print = True
    # print('board step', self.step)
    # print('num of ghost', len(list(self.my_ghost_ships)))
    # print('num of enemy_shipyards', len(list(self.enemy_shipyards)))
    # print('num of not target enemy_shipyards',
    # len(list(non_targeted_enemy_shipyards())))

    for ship in self.my_ghost_ships:
      found_enemy_yard = False
      _, min_dist_yard = self.find_nearest_shipyard(
          ship, non_targeted_enemy_shipyards())
      if min_dist_yard:
        self.ship_move_task(ship, min_dist_yard.cell, P_DESTORY_ENEMY_YARD)
        found_enemy_yard = True
        # print('sending ship', ship.id, "to", min_dist_yard.id, "at",
        # min_dist_yard.cell.position)

      # Convert ghost to farmer if no enemy shipyard found.
      if not found_enemy_yard:
        self.SHIP_ID_TO_TYPE[ship.id] = ShipType.FARMER_TYPE

    # if do_print:
    # print('num of ghost after', len(list(self.my_ghost_ships)))

  def send_ship_to_halite(self):
    """Ship that goes to halite."""
    for ship in self.my_idle_farmer_ships:
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
    next_move_positions = set()
    for ship in ships:
      self.take_move(ship, next_move_positions)
      next_move_positions.add(ship.next_cell.position)

  def execute(self):
    self.collect_game_info()

    self.convert_to_shipyard()

    self.spawn_ships()

    self.send_ship_to_enemy_shipyard()

    # TODO(wangfei): merge it into send_ship_to_halite()
    self.continue_mine_halite()
    self.send_ship_to_shipyard()

    # TODO: add priority for leaving if a ship is converted on yard.
    self.send_ship_to_halite()
    self.compute_ship_moves()

    # TODO: maybe no longer need it?
    self.collision_avoid()

  def spawn_ships(self):
    """Spawns farmer ships if we have enough money and no collision with my own
    ships."""
    board = self.board
    me = board.current_player

    num_ships = len(me.ship_ids)
    if num_ships >= MAX_SHIP_NUM:
      return

    num_ghost_ships = len(list(self.my_ghost_ships))
    num_enemy_yard = len(list(self.enemy_shipyards))
    if num_ships >= MAX_FARMER_SHIP_NUM and num_ghost_ships >= num_enemy_yard:
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
      # num_ships += 1
      # if num_ships >= MAX_FARMER_SHIP_NUM:
      # break


def agent(obs, config):
  board = Board(obs, config)

  # Init
  for cell in board.cells.values():
    cell.is_targetd = False
    cell.is_occupied = False

  strategy = ShipStrategy(board)
  strategy.execute()
  return board.current_player.next_actions
