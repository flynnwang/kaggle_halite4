#!/usr/bin/env python
"""
Plans to use match for ship next move assignment.


"""

import random
import timeit
import logging
from enum import Enum, auto
from collections import deque

import networkx as nx
import numpy as np
from kaggle_environments.envs.halite.helpers import *

MIN_WEIGHT = -99999

# If less than this value, Give up mining more halite from this cell.
CELL_STOP_COLLECTING_HALITE = 50.0
CELL_START_COLLECTING_HALITE = 80.0

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

# The factor is num_of_ships : num_of_shipyards
SHIP_TO_SHIYARD_FACTOR = 100

# To control the mining behaviour
MIN_HALITE_BEFORE_HOME = 100
MIN_HALITE_FACTOR = 3

MAX_FARMER_SHIP_NUM = 15
MAX_SHIP_NUM = 20

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class Timer:

  def __init__(self, logging_text=None):
    self._logging_text = logging_text
    self._start = None
    self._end = None
    self._interval = None

  def __enter__(self):
    self._start = timeit.default_timer()
    return self

  def __exit__(self, *args):
    self._end = timeit.default_timer()
    self._interval = self._end - self._start
    if self._logging_text is not None:
      logger.info("Took %.3f seconds for %s", self._interval,
                  self._logging_text)

  @property
  def interval(self):
    return self._interval


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


class ShipTask(Enum):

  UNKNOWN_TASK = auto()
  STAY_ON_CELL_TASK = auto()
  GO_TO_HALITE_TASK = auto()
  COLLECT_HALITE_TASK = auto()
  RETURN_TO_SHIPYARD_TASK = auto()
  DESTORY_ENEMY_YARD_TASK = auto()


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
    |task_type|: used to rank ship for moves.
  """

  SHIP_ID_TO_TYPE = {}

  def __init__(self, board):
    self.board = board
    self.me = board.current_player

    # Init halite cells
    self.halite_cells = []
    for cell in board.cells.values():
      if cell.halite > CELL_START_COLLECTING_HALITE:
        self.halite_cells.append(cell)

    # Default ship to stay on the same cell without assignment.
    # print('#', self.step)
    ships = self.me.ships
    has_enough_farmer = len(ships) > MAX_FARMER_SHIP_NUM

    for ship in ships:
      # print('ship', ship.id, 'at', ship.position, 'halite', ship.halite)
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.task_type = ShipTask.STAY_ON_CELL_TASK

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
  def add_ship_task(ship, target_cell: Cell, task_type: ShipTask):
    ship.has_assignment = True
    ship.target_cell = target_cell
    ship.target_cell.is_targetd = True
    ship.task_type = task_type

  @staticmethod
  def ship_stay(ship):
    ship.next_action = None
    ship.next_cell = ship.cell
    ship.next_cell.is_occupied = True
    if ship.cell.halite > 0:
      ship.cell.is_targetd = True

  def max_expected_return_cell(self, ship):
    growth = self.board.configuration.regen_rate + 1.0

    max_cell = None
    max_expected_return = 0
    for c in self.halite_cells:
      if c.is_targetd:
        continue

      expect_return = self.compute_expect_halite_return(ship, ship.position, c)
      if expect_return < 0:
        continue

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
      if (max_cell and max_cell.halite >= CELL_STOP_COLLECTING_HALITE and
          max_cell.position == ship.position):
        self.add_ship_task(ship, max_cell, ShipTask.COLLECT_HALITE_TASK)

  def send_ship_to_shipyard(self):
    """Ship goes back home after collected enough halite."""
    threshold = int(
        max(self.mean_halite_value * MIN_HALITE_FACTOR, MIN_HALITE_BEFORE_HOME))
    for ship in self.my_idle_farmer_ships:
      if ship.halite < threshold:
        continue

      # TODO: if too many ships are home, shall we wait?
      _, min_dist_yard = self.find_nearest_shipyard(ship, self.me.shipyards)
      if min_dist_yard:
        self.add_ship_task(ship, min_dist_yard.cell,
                           ShipTask.RETURN_TO_SHIPYARD_TASK)

  def compute_expect_halite_return(self, ship, position, target_cell):
    """Position maybe other than ship.position in case of computing moves"""
    growth = self.board.configuration.regen_rate + 1.0
    collect_rate = self.board.configuration.collect_rate
    board_size = self.board.configuration.size

    dist = manhattan_dist(position, target_cell.position, board_size)
    if target_cell.ship_id:
      # Halite will decrease if there is ship.
      expected_halite = target_cell.halite * ((1 - collect_rate)**dist)

      # Give up if my ship has more halite then enemy.
      if has_enemy_ship(target_cell, self.me):
        collectd = target_cell.halite - expected_halite
        enemy_halite = target_cell.ship.halite + collectd
        if ship.halite >= enemy_halite:
          return -1
    else:
      # Otherwise, halite will grow.
      expected_halite = min(target_cell.halite * (growth**dist),
                            self.board.configuration.max_cell_halite)
    return expected_halite / (dist + 1)

  def send_ship_to_halite(self):
    """Ship that goes to halite."""
    for ship in self.my_idle_farmer_ships:
      _, max_cell = self.max_expected_return_cell(ship)
      if max_cell:
        self.add_ship_task(ship, max_cell, ShipTask.GO_TO_HALITE_TASK)

  def send_ship_to_halite_v2(self):
    """Ship that goes to halite."""
    board_size = self.board.configuration.size

    g = nx.Graph()
    for ship in self.my_idle_farmer_ships:
      for cell in self.halite_cells:
        if cell.halite < CELL_START_COLLECTING_HALITE:
          continue

        cell_dist = manhattan_dist(ship.position, cell.position, board_size)
        if cell_dist > 10:
          continue

        wt = self.compute_expect_halite_return(ship, ship.position, cell)
        if wt < 0:
          continue
        g.add_edge(ship.id, cell.position, weight=int(wt * 100))

    matches = nx.algorithms.max_weight_matching(g, weight='weight')
    for ship_id, cell_pos in matches:
      # Assume ship id is str.
      if not isinstance(ship_id, str):
        ship_id, cell_pos = cell_pos, ship_id

      ship = self.board.ships[ship_id]
      max_cell = self.board[cell_pos]
      halite_return = self.compute_expect_halite_return(ship, ship.position,
                                                        max_cell)

      home_dist, min_dist_yard = self.find_nearest_shipyard(
          ship, self.me.shipyards)
      home_return = ship.halite / (home_dist + 1)

      if home_return > halite_return:
        self.add_ship_task(ship, min_dist_yard.cell,
                           ShipTask.RETURN_TO_SHIPYARD_TASK)
      else:
        self.add_ship_task(ship, max_cell, ShipTask.GO_TO_HALITE_TASK)

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
        self.add_ship_task(ship, min_dist_yard.cell,
                           ShipTask.DESTORY_ENEMY_YARD_TASK)
        found_enemy_yard = True
        # print('sending ship', ship.id, "to", min_dist_yard.id, "at",
        # min_dist_yard.cell.position)

      # Convert ghost to farmer if no enemy shipyard found.
      if not found_enemy_yard:
        self.SHIP_ID_TO_TYPE[ship.id] = ShipType.FARMER_TYPE

    # if do_print:
    # print('num of ghost after', len(list(self.my_ghost_ships)))

  def collision_check(self):
    ship_positions = {ship.position for ship in self.me.ships}
    assert len(ship_positions) == len(self.me.ships)

  def convert_to_shipyard(self):
    """Builds shipyard with a random ship if we have enough halite and ships."""
    # TODO: check on ship halite to convert to halite.
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
    """Computes ship moves to its target.

    Maximize total expected value.
    * prefer the move to the target (distance is shorter)
    * not prefer move into enemy with lower halite (or equal)
    * TODO: prefer cell which is near ally ships 
    """
    board_size = self.board.configuration.size
    spawn_cost = self.board.configuration.spawn_cost
    convert_cost = self.board.configuration.convert_cost
    collect_rate = self.board.configuration.collect_rate

    # Skip only convert ships.
    ships = [s for s in self.me.ships if not s.next_action]

    possible_moves = [
        Point(0, 0),
        Point(0, 1),
        Point(0, -1),
        Point(1, 0),
        Point(-1, 0)
    ]
    random.shuffle(possible_moves)

    def compute_weight(ship, next_position):
      target_cell = ship.target_cell
      next_cell = self.board[next_position]
      is_ghost_ship = self.SHIP_ID_TO_TYPE[ship.id] == ShipType.GHOST_TYPE

      dist = manhattan_dist(next_position, target_cell.position, board_size)
      ship_dist = manhattan_dist(ship.position, target_cell.position,
                                 board_size)

      # If stay at current location, prefer not stay...
      wt = ship_dist - dist
      if (ship.task_type == ShipTask.STAY_ON_CELL_TASK and
          ship.position == next_position):
        wt -= 10

      # If collecting halite
      if ((ship.task_type == ShipTask.GO_TO_HALITE_TASK or
           ship.task_type == ShipTask.COLLECT_HALITE_TASK) and
          target_cell.halite > 0):
        expect_return = self.compute_expect_halite_return(
            ship, next_position, ship.target_cell)
        if expect_return < 0:
          return MIN_WEIGHT
        wt += expect_return

      # If go back home
      if ship.task_type == ShipTask.RETURN_TO_SHIPYARD_TASK:
        wt += ship.halite / (dist + 1)

      # If goto enemy yard.
      if ship.task_type == ShipTask.DESTORY_ENEMY_YARD_TASK:
        # TODO: use what value as weight for destory enemy yard?
        wt += convert_cost / (dist + 1)

      # If next move to alley SPAWNING shipyard, skip
      yard = next_cell.shipyard
      if (yard and yard.player_id == self.me.id and
          yard.next_action == ShipyardAction.SPAWN):
        return MIN_WEIGHT

      # If there is an enemy in next_position with lower halite
      for nb_cell in get_neighbor_cells(next_cell, include_self=True):
        if has_enemy_ship(nb_cell, self.me):
          if (is_ghost_ship and nb_cell.ship.halite == ship.halite or
              nb_cell.ship.halite < ship.halite):
            wt -= spawn_cost

      assert wt != 0, (
          "weight for moving should not be zero: %s" % ship.task_type)
      return wt

    g = nx.Graph()
    for ship in ships:
      for move in possible_moves:
        next_position = ship.position + move
        wt = compute_weight(ship, next_position)
        if wt == MIN_WEIGHT:
          continue
        # print('   ship ', ship.id, 'to', next_position, 'target',
        # ship.target_cell.position, 'wt=', wt)
        g.add_edge(ship.id, next_position, weight=int(wt * 100))

    matches = nx.algorithms.max_weight_matching(
        g, maxcardinality=True, weight='weight')
    assert len(matches) == len(ships)
    # print('ships=', len(ships), self.board.step)
    for ship_id, position in matches:
      # Assume ship id is str.
      if not isinstance(ship_id, str):
        ship_id, position = position, ship_id

      ship = self.board.ships[ship_id]
      next_cell = self.board[position]
      # print(ship_id, 'at', ship.position, 'goto', position)

      move = position - ship.position
      ship.next_action = direction_to_ship_action(move)
      ship.next_cell = next_cell
      next_cell.is_occupied = True

  def execute(self):
    self.collect_game_info()

    self.convert_to_shipyard()

    self.spawn_ships()

    self.send_ship_to_enemy_shipyard()

    self.continue_mine_halite()
    self.send_ship_to_shipyard()
    self.send_ship_to_halite()
    self.compute_ship_moves()
    self.collision_check()

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
      # print('shipyard', shipyard.id, 'at', shipyard.position)

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
