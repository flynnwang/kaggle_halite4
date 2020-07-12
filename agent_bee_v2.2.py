#!/usr/bin/env python
"""
"""

import copy
import random
import timeit
import logging
from enum import Enum, auto

import networkx as nx
import numpy as np
from kaggle_environments.envs.halite.helpers import *

MIN_WEIGHT = -99999

BEGINNING_PHRASE_END_STEP = 40
NEAR_ENDING_PHRASE_STEP = 340
ENDING_PHRASE_STEP = 370

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

FUTURE_RATIO = 0.3

# Controls the number of ships.
EXPECT_SHIP_NUM = 20
MAX_SHIP_NUM = 30
MIN_FARMER_NUM = 9

# Threshold for attack enemy nearby my shipyard
TIGHT_ENEMY_SHIP_DEFEND_DIST = 5
LOOSE_ENEMY_SHIP_DEFEND_DIST = 7
AVOID_COLLIDE_RATIO = 0.6

MEAN_HALITE_OFFSET = 10

# Threshod used to send bomb to enemy shipyard
MIN_ENEMY_YARD_TO_MY_YARD = 6

POSSIBLE_MOVES = [
    Point(0, 0),
    Point(0, 1),
    Point(0, -1),
    Point(1, 0),
    Point(-1, 0)
]

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


def cargo(player):
  """Computes the cargo value for a player."""
  return sum([s.halite for s in player.ships], 0)


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


def direction_to_ship_action(position, next_position, board_size):
  if position == next_position:
    return None
  if (position + Point(0, 1)) % board_size == next_position:
    return ShipAction.NORTH
  if (position + Point(1, 0)) % board_size == next_position:
    return ShipAction.EAST
  if (position + Point(0, -1)) % board_size == next_position:
    return ShipAction.SOUTH
  if (position + Point(-1, 0)) % board_size == next_position:
    return ShipAction.WEST
  assert False, '%s, %s' % (position, next_position)


def make_move(position, move, board_size):
  return (position + move) % board_size


def get_neighbor_cells(cell, include_self=False):
  neighbor_cells = [cell] if include_self else []
  neighbor_cells.extend([cell.north, cell.south, cell.east, cell.west])
  return neighbor_cells


class ShipTask(Enum):

  UNKNOWN_TASK = auto()

  # Default task, to stay on current cell.
  STAY_ON_CELL_TASK = auto()

  # Send a ship to a halite.
  GOTO_HALITE_TASK = auto()

  # Continuing collecting halite on current cell.
  COLLECT_HALITE_TASK = auto()

  # Finish collecting halite and go back to shipyard.
  RETURN_TO_SHIPYARD_TASK = auto()

  # Send a ship to attack enemy's shipyard.
  DESTORY_ENEMY_YARD_TASK = auto()

  # Attack enemy ship from inside, near my shipyard.
  DESTORY_ENEMY_TASK_INNER = auto()

  # Attack enemy ship from outside, far from my shipyard.
  DESTORY_ENEMY_TASK_OUTER = auto()

  # Send the first ship to a location to build the initial shipyard.
  GOTO_INITIAL_YARD_POS_TASK = auto()

  # Make one ship stay on the shipyard to protect it from enemy next to it.
  GUARD_SHIPYARD_TASK = auto()


# cached values
HALITE_RETENSION_BY_DIST = []
HALITE_GROWTH_BY_DIST = []


class ShipStrategy:
  """Sends every ships to the nearest cell with halite.

  cell:
    |is_targetd|: prevent more than 1 alley ships choose the same halite.

  ship:
    |next_cell|: next cell location of the ship.
    |has_assignment|: has already has task assignment for this ship.
    |target_cell|: send ship to this cell, may be the same of current cell.
    |task_type|: used to rank ship for moves.
  """

  def __init__(self, board, simulation=False):
    self.board = board
    self.me = board.current_player
    self.cost_halite = 0
    self.simulation = simulation

    self.init_halite_cells()

    # Default ship to stay on the same cell without assignment.
    for ship in self.me.ships:
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.task_type = ShipTask.STAY_ON_CELL_TASK

    # init constants
    global HALITE_GROWTH_BY_DIST
    if not HALITE_GROWTH_BY_DIST:
      HALITE_GROWTH_BY_DIST = [
          self.growth_factor**d for d in range(self.c.size**2 + 1)
      ]

    global HALITE_RETENSION_BY_DIST
    if not HALITE_RETENSION_BY_DIST:
      HALITE_RETENSION_BY_DIST = [
          self.retension_rate_rate**d for d in range(self.c.size**2 + 1)
      ]
    # print(HALITE_GROWTH_BY_DIST, HALITE_RETENSION_BY_DIST)

  def init_halite_cells(self):
    MIN_STOP_COLLECTIONG_THRESHOLD = 10

    def keep_halite_value(cell):
      # Collect all halite of a cell during the end phrase.
      if self.step >= NEAR_ENDING_PHRASE_STEP:
        return MIN_STOP_COLLECTIONG_THRESHOLD

      # lower_bound = max(self.mean_halite_value * 0.3,
      # MIN_STOP_COLLECTIONG_THRESHOLD)
      threshold = MIN_STOP_COLLECTIONG_THRESHOLD
      # if self.step < BEGINNING_PHRASE_END_STEP:
      # threshold = max(self.mean_halite_value - 30, MIN_ENEMY_YARD_TO_MY_YARD)

      yard_dist, _ = self.get_nearest_home_yard(cell)
      if yard_dist <= self.home_grown_cell_dist:
        plus = max(self.num_ships - 20, 0) * 10
        # if self.num_ships >= 23 or self.num_shipyards >= 3:
        # plus += 50
        # if self.num_ships >= 28 and self.num_shipyards >= 3:
        # plus += 50
        threshold = 100 + plus
        if (BEGINNING_PHRASE_END_STEP < self.step < NEAR_ENDING_PHRASE_STEP and
            yard_dist <= self.tight_defend_dist() and self.num_ships >= 16):
          threshold = 200 + plus
          # if self.num_ships >= 16:
          # threshold += 50
          # if self.num_ships >= EXPECT_SHIP_NUM:
          # threshold += 50
          # if self.num_ships >= MAX_SHIP_NUM:
          # threshold = self.dist_to_expected_halite(yard_dist)

      threshold = min(threshold, 400)
      return threshold

    # Init halite cells
    self.halite_cells = []
    for cell in self.board.cells.values():
      cell.is_targetd = False
      if cell.halite > 0:
        self.halite_cells.append(cell)

    # Initialize covered cells by shipyards.
    self.covered_positions = set()
    for cell in self.halite_cells:
      min_dist, _ = self.get_nearest_home_yard(cell)
      if min_dist <= self.tight_defend_dist():
        self.covered_positions.add(cell.position)

    self.mean_halite_value = 0
    if self.halite_cells:
      halite_values = [c.halite for c in self.halite_cells]
      self.mean_halite_value = np.mean(halite_values)
      self.std_halite_value = np.std(halite_values)

    for cell in self.halite_cells:
      cell.keep_halite_value = keep_halite_value(cell)

  @property
  def c(self):
    return self.board.configuration

  @property
  def step(self):
    return self.board.step

  @property
  def growth_factor(self):
    return self.board.configuration.regen_rate + 1.0

  @property
  def retension_rate_rate(self):
    return 1. - self.c.collect_rate

  @property
  def num_ships(self):
    return len(self.me.ship_ids)

  @property
  def me_halite(self):
    return self.me.halite - self.cost_halite

  @property
  def num_shipyards(self):
    return len(self.me.shipyard_ids)

  @property
  def my_idle_ships(self):
    """All ships without task assignment."""
    for ship in self.me.ships:
      if ship.next_action or ship.has_assignment:
        continue
      yield ship

  @property
  def enemy_shipyards(self):
    for e in self.board.opponents:
      for y in e.shipyards:
        yield y

  @staticmethod
  def assign_task(ship, target_cell: Cell, task_type: ShipTask, enemy=None):
    """Add a task to a ship."""
    ship.has_assignment = True
    ship.target_cell = target_cell
    ship.task_type = task_type
    ship.target_cell.is_targetd = True
    ship.target_enemy = enemy

  def find_nearest_shipyard(self, cell: Cell, shipyards):
    position = cell.position

    min_dist = 99999
    min_dist_yard = None
    for y in shipyards:
      d = manhattan_dist(position, y.position, self.c.size)
      if d < min_dist:
        min_dist = d
        min_dist_yard = y
    return min_dist, min_dist_yard

  def get_nearest_home_yard(self, cell):
    if not hasattr(cell, 'home_yard_info'):
      cell.home_yard_info = self.find_nearest_shipyard(cell, self.me.shipyards)
    return cell.home_yard_info

  def get_nearest_enemy_yard(self, cell):
    if not hasattr(cell, 'enemy_yard_info'):
      cell.enemy_yard_info = self.find_nearest_shipyard(cell,
                                                        self.enemy_shipyards)
    return cell.enemy_yard_info

  def collect_game_info(self):

    # Computes neighbour cells mean halite values.
    # TODO: reuse
    def cell_to_yard_dist(cell):
      min_dist, _ = self.get_nearest_home_yard(cell)
      return min_dist

    self.mean_home_halite = 100
    home_cells = [
        cell.halite
        for cell in self.halite_cells
        if cell_to_yard_dist(cell) <= self.home_grown_cell_dist
    ]
    if home_cells:
      self.mean_home_halite = np.mean(home_cells)

    self.max_enemy_halite = -1
    self.max_enemy_id = None
    for p in self.board.opponents:
      if p.halite >= self.max_enemy_halite:
        self.max_enemy_halite = p.halite
        self.max_enemy_id = p.id

  def max_expected_return_cell(self, ship):
    max_cell = None
    max_expected_return = 0
    for cell in self.halite_cells:
      # Ignore targeted or cell with "little" halite.
      if cell.is_targetd or cell.halite < cell.keep_halite_value:
        continue

      # Do not go into enemy shipyard for halite.
      enemy_yard_dist, enemy_yard = self.get_nearest_enemy_yard(cell)
      if (enemy_yard and enemy_yard_dist <= 3):
        ally_yard_dist, alley_yard = self.get_nearest_home_yard(cell)
        if (alley_yard and enemy_yard_dist < ally_yard_dist):
          continue
        if alley_yard and ally_yard_dist <= self.home_grown_cell_dist:
          continue

      expected_return = self.compute_expect_halite_return(
          ship, ship.position, cell)
      if expected_return > max_expected_return:
        max_cell = cell
        max_expected_return = expected_return
    return max_expected_return, max_cell

  def continue_mine_halite(self):
    """Ship that stay on halite cell. We're trying to collect more halite from
    far away cells, but keep some margin on home grown cells."""

    for ship in self.my_idle_ships:
      _, max_cell = self.max_expected_return_cell(ship)
      if max_cell and max_cell.position == ship.position:
        self.assign_task(ship, max_cell, ShipTask.COLLECT_HALITE_TASK)

  def send_ship_to_halite(self):
    """Ship that goes to halite."""
    for ship in self.my_idle_ships:
      _, max_cell = self.max_expected_return_cell(ship)
      if max_cell:
        self.assign_task(ship, max_cell, ShipTask.GOTO_HALITE_TASK)

  def send_ship_to_home_yard(self):
    """Ship goes back home after collected enough halite."""

    # TODO(wangfei): if tracked by enemy, also send back home.

    # NOTE: keep this value somehow bigger to reduce waste of time.
    # MIN_HALITE_BEFORE_HOME = 150 # 23000
    # TODO(wangfei): test this value, it's important.
    MIN_HALITE_BEFORE_HOME = 100

    def return_home_yard_halite_threshold():
      # This not work, because it limits the ship spawn with initial halite.
      # if self.step <= BEGINNING_PHRASE_END_STEP:
      # return 300

      if self.step >= NEAR_ENDING_PHRASE_STEP:
        return 400
      # return self.mean_home_halite + 10
      # return max(self.mean_home_halite, MIN_HALITE_BEFORE_HOME)
      # return max(self.mean_halite_value + MEAN_HALITE_OFFSET,
      # MIN_HALITE_BEFORE_HOME)
      # return (self.mean_halite_value * self.c.collect_rate * 3)
      # return self.mean_halite_value

      return int(max(self.mean_halite_value, MIN_HALITE_BEFORE_HOME))

    # print('***min_h=', return_home_yard_halite_threshold())
    for ship in self.my_idle_ships:
      if ship.halite < return_home_yard_halite_threshold():
        continue

      _, min_dist_yard = self.get_nearest_home_yard(ship.cell)
      if min_dist_yard:
        self.assign_task(ship, min_dist_yard.cell,
                         ShipTask.RETURN_TO_SHIPYARD_TASK)

  def compute_expect_halite_return(self, ship, position, target_cell):
    """Position maybe other than ship.position in case of computing moves"""
    dist = manhattan_dist(position, target_cell.position, self.c.size)
    if target_cell.ship_id:
      # Halite will decrease if there is ship.
      expected_halite = target_cell.halite * HALITE_RETENSION_BY_DIST[dist]

      # Give up if my ship has more halite then enemy.
      if has_enemy_ship(target_cell, self.me):
        collectd = target_cell.halite - expected_halite
        enemy_halite = target_cell.ship.halite + collectd
        if ship and ship.halite >= enemy_halite:
          return -1
    else:
      # Otherwise, halite will grow.
      expected_halite = min(target_cell.halite * HALITE_GROWTH_BY_DIST[dist],
                            self.c.max_cell_halite)
    return expected_halite / (dist + 1)

  # TODO(wangfei): try this again later.
  def send_ship_to_halite_v2(self):
    """Ship that goes to halite."""
    board_size = self.board.configuration.size

    g = nx.Graph()
    for ship in self.my_idle_ships:
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

      home_dist, min_dist_yard = self.get_nearest_home_yard(ship.cell)
      home_return = ship.halite / (home_dist + 1)

      if home_return > halite_return:
        self.assign_task(ship, min_dist_yard.cell,
                         ShipTask.RETURN_TO_SHIPYARD_TASK)
      else:
        self.assign_task(ship, max_cell, ShipTask.GOTO_HALITE_TASK)

  def bomb_enemy_shipyard(self):
    """Having enough farmers, let's send ghost to enemy shipyard."""

    def max_bomb_dist():
      # Don't use bomb if ship group is small.
      if self.num_ships <= 25:
        return 0

      # If having enough money and halite.
      # if (self.num_ships >= MAX_SHIP_NUM and
      if (self.num_ships >= 30 and
          self.me.halite > self.max_enemy_halite + self.c.spawn_cost * 2):
        return 999

      # Only attack nearby enemy yard.
      return MIN_ENEMY_YARD_TO_MY_YARD

    def is_near_my_shipyard(enemy_yard):
      for yard in self.me.shipyards:
        dist = manhattan_dist(yard.position, enemy_yard.position, self.c.size)
        if dist <= max_bomb_dist():
          return True
      return False

    def non_targeted_enemy_shipyards():
      for y in self.enemy_shipyards:
        if y.cell.is_targetd:
          continue
        if not is_near_my_shipyard(y):
          continue
        yield y

    def select_bomb_ship(enemy_yard):
      min_dist = 99999
      bomb_ship = None
      for ship in self.my_idle_ships:
        # Don't send halite to enemy.
        if ship.halite > 0:
          continue
        dist = manhattan_dist(enemy_yard.position, ship.position, self.c.size)
        if dist < min_dist:
          min_dist = dist
          bomb_ship = ship
      return min_dist, bomb_ship, enemy_yard

    if self.step < BEGINNING_PHRASE_END_STEP:
      return

    enemy_shipyards = (
        select_bomb_ship(y) for y in non_targeted_enemy_shipyards())
    enemy_shipyards = [x for x in enemy_shipyards if x[1]]
    enemy_shipyards.sort(key=lambda x: x[0])
    for _, bomb_ship, enemy_yard in enemy_shipyards:
      self.assign_task(bomb_ship, enemy_yard.cell,
                       ShipTask.DESTORY_ENEMY_YARD_TASK)

      # One bomb at a time
      break

  def tight_defend_dist(self):
    # return 4 + max((self.num_ships - 15) // 5, 0)
    return TIGHT_ENEMY_SHIP_DEFEND_DIST

  def loose_defend_dist(self):
    # return self.tight_defend_dist() + 2
    return LOOSE_ENEMY_SHIP_DEFEND_DIST

  def shipyard_defend_dist(self):
    return 3 if len(self.me.shipyard_ids) > 1 else 4

  @property
  def home_grown_cell_dist(self):
    return self.tight_defend_dist() + 1

  def dist_to_expected_halite(self, dist):
    """Start collect halite if its halite reach this value."""
    if dist == 0:
      return 0
    h = 0
    if dist <= 2:
      return 400
    elif dist <= 3:
      h = 350
    elif dist <= 4:
      h = 300
    elif dist <= self.home_grown_cell_dist:
      h = 150
    else:
      h = 30
    h += 20 * max((self.num_ships - 15), 0)
    return min(h, 450)

  @property
  def is_beginning_phrase(self):
    return self.step <= BEGINNING_PHRASE_END_STEP

  def attack_enemy_ship(self, has_budget=True):
    """Send ship to enemy to protect my shipyard."""

    def all_enemy_ships(defend_distance):
      for opp in self.board.opponents:
        for s in opp.ships:
          min_dist, min_yard = self.get_nearest_home_yard(s.cell)
          if min_yard and min_dist <= defend_distance:
            yield s, min_dist, min_yard

    def defend_ship_ratio():
      if self.num_ships <= 10:
        ratio = 0.3
      elif self.num_ships <= 13:
        ratio = 0.4
      elif self.num_ships <= 17:
        ratio = 0.45
      elif self.num_ships <= 20:
        ratio = 0.5
      elif self.num_ships <= 30:
        ratio = 0.55
      ratio = 0.6
      return ratio

    def min_farmer_num():
      # Use all ships for halite collection.
      if (self.step >= NEAR_ENDING_PHRASE_STEP or
          self.step < BEGINNING_PHRASE_END_STEP):
        return max(self.num_ships - self.num_shipyards, 0)

      farmer_ratio = 1 - defend_ship_ratio()
      return max(MIN_FARMER_NUM,
                 int(np.round(self.num_ships * defend_ship_ratio())))

    def max_defend_ship_num():
      return 3

    enemy_ships = list(all_enemy_ships(self.loose_defend_dist()))
    enemy_ships.sort(key=lambda x: (x[1], -x[0].halite))
    ship_budget = self.num_ships
    if has_budget:
      ship_budget -= min_farmer_num()

    for enemy, enemy_to_defend_yard_dist, defend_yard in enemy_ships:
      # Skip enemy for the second round of attack if targeted.
      if enemy.cell.is_targetd:
        continue

      def dist_to_enemy(ship):
        return manhattan_dist(ship.position, enemy.position, self.c.size)

      def dist_to_defend_yard(ship):
        return manhattan_dist(ship.position, defend_yard.position, self.c.size)

      def get_target_cell(ship, offset_dist):
        min_dist = 999
        min_dist_cell = None
        for cell in get_neighbor_cells(enemy.cell):
          dist = manhattan_dist(cell.position, defend_yard.position,
                                self.c.size)
          if dist == enemy_to_defend_yard_dist + offset_dist:
            d = manhattan_dist(ship.position, cell.position, self.c.size)
            if d < min_dist:
              min_dist = d
              min_dist_cell = cell
        assert min_dist_cell
        return min_dist_cell

      # Protect shipyard
      if (ship_budget > 0 and not defend_yard.cell.is_targetd and
          enemy.halite <= 10 and
          enemy_to_defend_yard_dist <= self.shipyard_defend_dist()):
        ships = [
            s for s in self.my_idle_ships
            if (dist_to_defend_yard(s) < enemy_to_defend_yard_dist or
                (dist_to_defend_yard(s) == enemy_to_defend_yard_dist and
                 enemy.halite > 0 and s.halite < enemy.halite or
                 enemy.halite == 0 and s.halite <= enemy.halite))
        ]
        ships = sorted(ships, key=dist_to_defend_yard)
        for ship in ships[:min(1, ship_budget)]:
          self.assign_task(ship, defend_yard.cell, ShipTask.GUARD_SHIPYARD_TASK,
                           enemy)
          # print('guide task: ', ship.position, defend_yard.position)
          ship_budget -= 1

      # Skip attack enemy if not having enough ships.
      if (self.step < BEGINNING_PHRASE_END_STEP and
          self.step > NEAR_ENDING_PHRASE_STEP):
        continue

      # send destory ship from inner.
      if (ship_budget > 0 and
          enemy_to_defend_yard_dist <= self.tight_defend_dist()):
        ships = [
            s for s in self.my_idle_ships
            if ((enemy.halite > 0 and s.halite < enemy.halite or
                 enemy.halite == 0 and s.halite == 0) and
                dist_to_defend_yard(s) <= enemy_to_defend_yard_dist)
        ]
        ships = sorted(ships, key=dist_to_enemy)
        for ship in ships[:min(max_defend_ship_num(), ship_budget)]:
          target_cell = get_target_cell(ship, offset_dist=1)
          self.assign_task(ship, target_cell, ShipTask.DESTORY_ENEMY_TASK_INNER,
                           enemy)
          ship_budget -= 1

      # send destory ship from outer
      if ship_budget > 0:
        ships = [
            s for s in self.my_idle_ships
            if (s.halite < enemy.halite and
                dist_to_defend_yard(s) >= enemy_to_defend_yard_dist)
        ]
        ships.sort(key=dist_to_enemy)
        for ship in ships[:min(ship_budget, max_defend_ship_num())]:
          target_cell = get_target_cell(ship, offset_dist=+1)
          self.assign_task(ship, target_cell, ShipTask.DESTORY_ENEMY_TASK_OUTER,
                           enemy)
          ship_budget -= 1

      if ship_budget <= 0:
        break

  def collision_check(self):
    ships = [
        ship for ship in self.me.ships if ship.next_action != ShipAction.CONVERT
    ]
    ship_positions = {ship.next_cell.position for ship in ships}
    if len(ship_positions) != len(ships):
      print('Total ships', len(self.me.ships), 'step', self.board.step)
      for ship in self.me.ships:
        print(ship.id, 'at', ship.position, 'action', ship.next_action,
              'next_cell', ship.next_cell.position, 'task', ship.task_type)
    assert len(ship_positions) == len(ships)

  def convert_to_shipyard(self):
    """Builds shipyard to maximize the total number of halite covered within
    |home_grown_cell_dist|."""
    MAX_SHIPYARD_NUM = 5
    MIN_NEXT_YARD_DIST = 4
    MAX_NEXT_YARD_DIST = 6

    # No ship left.
    if not self.num_ships:
      return

    def max_shipyard_num():
      # Do not spawn too much shipyard at last.
      if self.num_ships >= 26:
        return min(3 + max((self.num_ships - 24) // 6, 0), MAX_SHIPYARD_NUM)
      if self.num_ships >= 25:
        return 3
      if self.num_ships >= 22:
        return 2
      # < 22
      return 1

    # Reach max shipyard num.
    if self.num_shipyards >= max_shipyard_num():
      return

    def convert_threshold():
      threshold = MIN_HALITE_TO_BUILD_SHIPYARD
      if (self.num_shipyards == 0 or
          self.board.step <= BEGINNING_PHRASE_END_STEP):
        threshold = self.c.convert_cost
      return max(self.c.convert_cost, threshold)

    def has_enough_halite(ship):
      return ship.halite + self.me.halite >= convert_threshold()

    def not_on_shipyard(ship):
      return ship.cell.shipyard_id is None

    def within_predefined_range(ship):
      if not self.me.shipyard_ids:
        return True
      # Put it near one of the nearest shipyard.
      min_dist, min_yard = self.get_nearest_home_yard(ship.cell)
      return min_yard and (MIN_NEXT_YARD_DIST <= min_dist <= MAX_NEXT_YARD_DIST)

    def score_ship_position(ship):
      # Maximize the total value of halite cells when converting ship.
      total_halite = 0
      total_cell = 0
      for cell in self.halite_cells:
        dist = manhattan_dist(ship.position, cell.position, self.c.size)
        if (dist <= self.tight_defend_dist() and
            cell.position not in self.covered_positions and
            cell.position != ship.position):
          total_halite += cell.halite
          total_cell += 1
      return total_cell, total_halite

    ship_scores = [(score_ship_position(s), s)
                   for s in self.me.ships
                   if (has_enough_halite(s) and not_on_shipyard(s) and
                       within_predefined_range(s))]
    ship_scores.sort(key=lambda x: x[0], reverse=True)
    for _, ship in ship_scores:
      self.cost_halite += (self.c.convert_cost - ship.halite)
      ship.next_action = ShipAction.CONVERT
      ship.has_assignment = True
      ship.cell.is_targetd = True

      # One shipyard at a time.
      break

  def compute_ship_moves(self):
    """Computes ship moves to its target.

    Maximize total expected value.
    * prefer the move to the target (distance is shorter)
    * not prefer move into enemy with lower halite (or equal)
    """
    spawn_cost = self.board.configuration.spawn_cost
    convert_cost = self.board.configuration.convert_cost
    collect_rate = self.board.configuration.collect_rate

    # Skip only convert ships.
    ships = [s for s in self.me.ships if not s.next_action]

    def compute_weight(ship, next_position):
      ignore_neighbour_cell_enemy = False
      target_cell = ship.target_cell
      next_cell = self.board[next_position]

      # If next move to alley SPAWNING shipyard, skip
      yard = next_cell.shipyard
      if (yard and yard.player_id == self.me.id and
          yard.next_action == ShipyardAction.SPAWN):
        return MIN_WEIGHT

      # If stay at current location, prefer not stay...
      dist = manhattan_dist(next_position, target_cell.position, self.c.size)
      ship_dist = manhattan_dist(ship.position, target_cell.position,
                                 self.c.size)
      wt = ship_dist - dist
      if (ship.task_type == ShipTask.STAY_ON_CELL_TASK and
          ship.position == next_position):
        wt -= 10

      # If collecting halite
      if ((ship.task_type == ShipTask.GOTO_HALITE_TASK or
           ship.task_type == ShipTask.COLLECT_HALITE_TASK) and
          target_cell.halite > 0):
        expect_return = self.compute_expect_halite_return(
            ship, next_position, target_cell)
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

      if (ship.task_type == ShipTask.DESTORY_ENEMY_TASK_OUTER or
          ship.task_type == ShipTask.DESTORY_ENEMY_TASK_INNER):
        wt += convert_cost / (dist + 1)
        enemy = ship.target_enemy
        enemy_dist = manhattan_dist(next_position, enemy.position, self.c.size)
        wt += enemy.halite / (enemy_dist + 1)

      if ship.task_type == ShipTask.GUARD_SHIPYARD_TASK:
        wt += 1 / (dist + 1)
        ignore_neighbour_cell_enemy = True

      def move_away_from_enemy(enemy, ship):
        """Only collide with enemy if my ship does not have any cargo."""
        if ship.halite > enemy.halite:
          return True
        # when <= enemy.halite
        if ship.halite > 0 and enemy.halite == ship.halite:
          return True
        # enemy.halite == ship.halite
        return random.random() < AVOID_COLLIDE_RATIO

      # If there is an enemy in next_position with lower halite
      if has_enemy_ship(next_cell, self.me):
        # If there is an enemy sitting on its shipyard, collide with him.
        if (ship.task_type == ShipTask.DESTORY_ENEMY_YARD_TASK and
            next_cell.position == target_cell.position):
          pass
        elif move_away_from_enemy(next_cell.ship, ship):
          wt -= (spawn_cost + ship.halite)

      # If there is an enemy in neighbor next_position with lower halite
      if not ignore_neighbour_cell_enemy:
        for nb_cell in get_neighbor_cells(next_cell):
          if has_enemy_ship(nb_cell, self.me):
            if move_away_from_enemy(nb_cell.ship, ship):
              wt -= (spawn_cost + ship.halite)
      return wt

    g = nx.Graph()
    for ship in ships:
      random.shuffle(POSSIBLE_MOVES)
      for move in POSSIBLE_MOVES:
        next_position = make_move(ship.position, move, self.c.size)
        wt = compute_weight(ship, next_position)
        if wt == MIN_WEIGHT:
          continue
        wt = int(wt * 100)
        g.add_edge(ship.id, next_position, weight=(wt if wt != 0 else -1))
    matches = nx.algorithms.max_weight_matching(
        g, maxcardinality=True, weight='weight')
    assert len(matches) == len(ships)

    for ship_id, next_position in matches:
      # Assume ship id is str.
      if not isinstance(ship_id, str):
        ship_id, next_position = next_position, ship_id

      ship = self.board.ships[ship_id]
      ship.next_cell = self.board[next_position]
      ship.next_action = direction_to_ship_action(ship.position, next_position,
                                                  self.c.size)
      # print(ship_id, 'at', ship.position, 'goto', next_position)

  def spawn_ships(self):
    """Spawns farmer ships if we have enough money and no collision with my own
    ships."""

    def max_ship_num():
      more_ships = max(0, (self.me_halite - 4000) // 1200)
      return min(50, MAX_SHIP_NUM + more_ships)

    def spawn(yard):
      self.cost_halite += self.c.spawn_cost
      yard.next_action = ShipyardAction.SPAWN

    def spawn_threshold():
      if self.step <= BEGINNING_PHRASE_END_STEP:
        return self.c.spawn_cost
      return MIN_HALITE_TO_BUILD_SHIP

    # Too many ships.
    mx = max_ship_num()
    print('mx=', mx)
    if self.num_ships >= max_ship_num():
      return

    # No more ships after ending.
    if self.num_ships >= 3 and self.step >= NEAR_ENDING_PHRASE_STEP:
      return

    shipyards = self.me.shipyards
    random.shuffle(shipyards)
    for shipyard in shipyards:
      # Only skip for the case where I have any ship.
      if self.num_ships and self.me_halite < spawn_threshold():
        continue

      spawn(shipyard)
      # One ship at a time
      break

  def final_stage_back_to_shipyard(self):
    MARGIN_STEPS = self.num_ships
    MIN_HALITE_TO_YARD = 10

    def ship_and_dist_to_yard():
      for ship in self.my_idle_ships:
        if ship.halite <= MIN_HALITE_TO_YARD:
          continue
        _, yard = self.get_nearest_home_yard(ship.cell)
        if yard:
          dist = manhattan_dist(ship.position, yard.position, self.c.size)
          yield dist, ship, yard

    if not self.me.shipyard_ids:
      return

    ship_dists = list(ship_and_dist_to_yard())
    if not ship_dists:
      return

    for min_dist, ship, min_dist_yard in ship_dists:
      if self.step + min_dist + MARGIN_STEPS > self.c.episode_steps:
        self.assign_task(ship, min_dist_yard.cell,
                         ShipTask.RETURN_TO_SHIPYARD_TASK)

  initial_shipyard_set = False
  initial_yard_position = None
  initial_ship_position = None

  def convert_first_shipyard(self):
    """Strategy for convert the first shipyard."""
    assert self.num_ships == 1, self.num_ships

    ship = self.me.ships[0]
    if not self.initial_ship_position:
      ShipStrategy.initial_ship_position = ship.position

    def simulate_initial_shipyard_position(position, max_steps=20):
      """Simulate returns without enemy for some steps."""
      ShipStrategy.initial_yard_position = position
      board = copy.deepcopy(self.board)
      for i in range(max_steps):
        strategy = ShipStrategy(board, simulation=True)
        strategy.execute()  # simulate for one step
        board = board.next()
        # strategy = ShipStrategy(board, simulation=False)
        # print('step=%s (b.step=%s), h=%s, s=%s, y=%s' %
        # (i, board.step, strategy.me.halite, strategy.num_ships,
        # strategy.num_shipyards))
      strategy = ShipStrategy(board, simulation=False)
      return (strategy.me.halite + strategy.num_ships * 500 +
              strategy.num_shipyards * 500 - 5000)

    def expected_coverted_halite(candidate_cell):
      expected_halite = 0
      current_halite = 0
      num_halite_cells = 0
      for cell in self.halite_cells:
        # shipyard will destory the halite under it.
        if candidate_cell.position == cell.position:
          continue

        dist = manhattan_dist(candidate_cell.position, cell.position,
                              self.c.size)
        if dist <= self.home_grown_cell_dist and cell.halite > 0:
          expected_halite += self.compute_expect_halite_return(
              None, ShipStrategy.initial_ship_position, cell)
          current_halite += cell.halite
          num_halite_cells += 1

      # simulte_halite = simulate_initial_shipyard_position(
      # candidate_cell.position)
      simulte_halite = 0
      print(candidate_cell.position, "expexted=",
            expected_halite, 'current=', current_halite, 'simulate=',
            int(simulte_halite), 'n=', num_halite_cells)
      # return score
      # Use current halite, because it will impact the initial ship building.
      # return (expected_halite, current_halite)
      return (simulte_halite, expected_halite, current_halite)

    def get_coord_range(v):
      DELTA = 2
      MARGIN = 2
      if v == 5:
        v_min, v_max = MARGIN, 5 + DELTA
      else:
        v_min, v_max = 15 - DELTA, 20 - MARGIN
      return v_min, v_max

    def select_initial_cell():
      position = ship.position
      x_min, x_max = get_coord_range(position.x)
      y_min, y_max = get_coord_range(position.y)
      for cell in self.board.cells.values():
        cell_pos = cell.position
        if (x_min <= cell_pos.x <= x_max and y_min <= cell_pos.y <= y_max):
          yield expected_coverted_halite(cell), cell

    if not self.initial_yard_position:
      candidate_cells = list(select_initial_cell())
      if candidate_cells:
        candidate_cells.sort(key=lambda x: x[0], reverse=True)
        value, yard_cell = candidate_cells[0]
        ShipStrategy.initial_yard_position = yard_cell.position
        print("Ship initial:", self.initial_ship_position, 'dist=',
              manhattan_dist(self.initial_ship_position,
                             self.initial_yard_position, self.c.size),
              'selected yard position:', self.initial_yard_position, 'value=',
              value)

    if ship.position == self.initial_yard_position:
      ShipStrategy.initial_shipyard_set = True
      ship.next_action = ShipAction.CONVERT
      ship.has_assignment = True
      ship.cell.is_targetd = True
    else:
      self.assign_task(ship, self.board[self.initial_yard_position],
                       ShipTask.GOTO_INITIAL_YARD_POS_TASK)

  def spawn_if_shipyard_in_danger(self):
    """Spawn ship if enemy nearby my shipyard and no ship's next_cell on this
    shipyard."""
    if self.step >= NEAR_ENDING_PHRASE_STEP:
      return
    ship_next_positions = {
        ship.next_cell.position
        for ship in self.me.ships
        if ship.next_action != ShipAction.CONVERT
    }

    def is_shipyard_in_danger(yard):
      # If there is one of my ship will be the on yard in the next round.
      if yard.position in ship_next_positions:
        return False
      for nb_cell in get_neighbor_cells(yard.cell):
        if has_enemy_ship(nb_cell, self.me):
          return True
      return False

    def spawn(yard):
      self.cost_halite += self.c.spawn_cost
      yard.next_action = ShipyardAction.SPAWN

    for yard in self.me.shipyards:
      # Skip shipyard already has action.
      if yard.next_action:
        continue
      if is_shipyard_in_danger(yard) and self.me_halite >= self.c.spawn_cost:
        print('spawn for danger: y=', yard.position, 'in_danger=',
              is_shipyard_in_danger(yard))
        spawn(yard)

  def print_info(self):

    def mean_cargo(player):
      num_ships = len(player.ship_ids)
      if num_ships == 0:
        return 0
      return int(cargo(player) / num_ships)

    o = sorted(self.board.opponents, key=lambda x: -(len(x.ship_ids)))[0]
    print('#', self.board.step, 'h(m=%s, s=%s)' % (int(self.mean_halite_value),
                                                   int(self.std_halite_value)),
          'yd=(m=%s, hc=%s)' % (int(self.mean_home_halite),
                                len(self.covered_positions)),
          'me(s=%s, y=%s, h=%s, c=%s, mc=%s)' % (self.num_ships,
                                                 len(self.me.shipyard_ids),
                                                 self.me_halite, cargo(self.me),
                                                 mean_cargo(self.me)),
          'e[%s](s=%s, h=%s, c=%s, mc=%s)' % (o.id, len(o.ships), o.halite,
                                              cargo(o), mean_cargo(o)))

  def execute(self):
    self.collect_game_info()

    if self.initial_shipyard_set:
      self.convert_to_shipyard()
      self.spawn_ships()

      self.bomb_enemy_shipyard()
      self.attack_enemy_ship()

      self.final_stage_back_to_shipyard()
      self.continue_mine_halite()
      self.send_ship_to_home_yard()
      self.send_ship_to_halite()

      # print('attack later=', len(list(self.my_idle_ships)))
      # Attack enemy if no task assigned
      self.attack_enemy_ship(has_budget=False)
    else:
      self.convert_first_shipyard()

    self.compute_ship_moves()
    self.spawn_if_shipyard_in_danger()
    if not self.simulation:
      self.print_info()


@board_agent
def agent(board):
  ShipStrategy(board).execute()
