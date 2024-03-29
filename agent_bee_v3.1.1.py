#!/usr/bin/env python
"""
v3.1.1 <- v3.1.0

* Use actual halite weight when convert shipyard.
* ATTACK_PER_ENEMY = 5
* SHIPYARD_DUPLICATE_NUM = 5

Tournament - ID: mzeKJO, Name: Your Halite 4 Trueskill Ladder | Dimension - ID: 81QjB8, Name: Halite 4 Dimension
Status: running | Competitors: 5 | Rank System: trueskill

Total Matches: 178 | Matches Queued: 59
bee v3.0                       | b8b48fA1SJm9   | 26.7642199      | μ=28.908, σ=0.714  | 139
bee v1.8                       | VACnTTkbobrZ   | 25.4959476      | μ=27.592, σ=0.699  | 136
bee v3.1.1                     | QiG692R5QtdC   | 24.3771178      | μ=26.483, σ=0.702  | 126
optimus_mining                 | UWGDiTKKWfEz   | 20.3328768      | μ=22.432, σ=0.700  | 140
c40                            | VaV4WfhdxChv   | 17.4870404      | μ=19.672, σ=0.728  | 171
"""

import copy
import random
import timeit
import logging
from collections import Counter
from enum import Enum, auto

import networkx as nx
import numpy as np
import scipy.optimize
from kaggle_environments.envs.halite.helpers import *

MIN_WEIGHT = -99999

BEGINNING_PHRASE_END_STEP = 40
NEAR_ENDING_PHRASE_STEP = 340
ENDING_PHRASE_STEP = 370

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

# Controls the number of ships.
EXPECT_SHIP_NUM = 20
MAX_SHIP_NUM = 25
MIN_FARMER_NUM = 9

# Threshold for attack enemy nearby my shipyard
TIGHT_ENEMY_SHIP_DEFEND_DIST = 5
LOOSE_ENEMY_SHIP_DEFEND_DIST = 7
AVOID_COLLIDE_RATIO = 0.6

# Threshod used to send bomb to enemy shipyard
MIN_ENEMY_YARD_TO_MY_YARD = 7

POSSIBLE_MOVES = [
    Point(0, 0),
    Point(0, 1),
    Point(0, -1),
    Point(1, 0),
    Point(-1, 0)
]

TURNS_OPTIMAL = np.array(
    [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7,
      8], [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7,
           7], [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
     [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6,
      6], [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
           6], [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4,
      4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3,
           3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def get_quadrant(p: Point):
  if p.x > 0 and p.y >= 0:
    return 1
  if p.x <= 0 and p.y > 0:
    return 2
  if p.x < 0 and p.y <= 0:
    return 3
  if p.x >= 0 and p.y < 0:
    return 4
  assert p == Point(0, 0), "not exist quadrant: %s %s" % (p.x, p.y)
  return 0


def optimal_mining_steps(C, H, rt_travel):
  # How many turns should we plan on mining?
  # C=carried halite, H=halite in square, rt_travel=steps to square and back to shipyard
  if C == 0:
    ch = 0
  elif H == 0:
    ch = TURNS_OPTIMAL.shape[0] - 1  # ?
  else:
    ch = int(np.log(C / H) * 2.5 + 5.5)
    ch = np.clip(ch, 0, TURNS_OPTIMAL.shape[0] - 1)
  rt_travel = int(np.clip(rt_travel, 0, TURNS_OPTIMAL.shape[1] - 1))
  return TURNS_OPTIMAL[ch, rt_travel]


logging.basicConfig(level=logging.INFO)
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
  STAY = auto()

  # Send a ship to a halite.
  GOTO_HALITE = auto()

  # Continuing collecting halite on current cell.
  COLLECT = auto()

  # Finish collecting halite and go back to shipyard.
  RETURN = auto()

  # Send a ship to attack enemy's shipyard.
  ATTACK_SHIPYARD = auto()

  # Attack enemy ship.
  ATTACK_SHIP = auto()

  # Send the first ship to a location to build the initial shipyard.
  INITIAL_SHIPYARD = auto()

  # Make one ship stay on the shipyard to protect it from enemy next to it.
  GUARD_SHIPYARD = auto()


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
    self.halite_ratio = -1
    self.num_home_halite_cells = 0
    self.simulation = simulation

    self.init_halite_cells()

    # Default ship to stay on the same cell without assignment.
    for ship in self.me.ships:
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.task_type = ShipTask.STAY

  def init_halite_cells(self):
    MIN_STOP_COLLECTIONG_THRESHOLD = 0

    def keep_halite_value(cell):
      threshold = MIN_STOP_COLLECTIONG_THRESHOLD

      if self.step >= NEAR_ENDING_PHRASE_STEP:
        return MIN_STOP_COLLECTIONG_THRESHOLD

      yard_dist, _ = self.get_nearest_home_yard(cell)
      if yard_dist <= self.home_grown_cell_dist:
        threshold = 100
        # plus = max(self.num_ships - 20, 0) * 10
        # threshold = 100 + plus
      threshold = min(300, threshold)

      # Do not go into enemy shipyard for halite.
      enemy_yard_dist, enemy_yard = self.find_nearest(cell, self.enemy_shipyards)
      if (enemy_yard and enemy_yard_dist <= 5):
        ally_yard_dist, alley_yard = self.find_nearest(cell, self.me.shipyards)
        if (alley_yard and enemy_yard_dist < ally_yard_dist):
          # if the cell is nearer to the enemy yard.
          return 999

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
      if min_dist <= self.home_grown_cell_dist:
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

  @property
  def enemy_ships(self):
    for e in self.board.opponents:
      for s in e.ships:
        yield s

  @staticmethod
  def assign_task(ship, target_cell: Cell, task_type: ShipTask, enemy=None):
    """Add a task to a ship."""
    ship.has_assignment = True
    ship.target_cell = target_cell
    ship.task_type = task_type
    ship.target_cell.is_targetd = True
    ship.target_enemy = enemy

  def find_nearest(self, cell: Cell, shipyards):
    min_dist = 99999
    min_dist_yard = None
    for y in shipyards:
      d = manhattan_dist(cell.position, y.position, self.c.size)
      if d < min_dist:
        min_dist = d
        min_dist_yard = y
    return min_dist, min_dist_yard

  def get_nearest_home_yard(self, cell):
    if not hasattr(cell, 'home_yard_info'):
      cell.home_yard_info = self.find_nearest(cell, self.me.shipyards)
    return cell.home_yard_info

  def get_nearest_enemy_yard(self, cell):
    if not hasattr(cell, 'enemy_yard_info'):
      cell.enemy_yard_info = self.find_nearest(cell,
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

    # Player info
    self.me.total_halite = self.me.halite + cargo(self.me)

    self.max_enemy_halite = -1
    self.max_enemy = None
    for p in self.board.opponents:
      p.total_halite = p.halite + cargo(p)
      if p.total_halite >= self.max_enemy_halite:
        self.max_enemy_halite = p.halite
        self.max_enemy= p

  def bomb_enemy_shipyard(self):
    """Having enough farmers, let's send ghost to enemy shipyard."""
    def estimate_halite(player):
      h = player.halite
      s = len(player.ship_ids) * self.c.spawn_cost
      return h + s

    def max_bomb_dist():
      # Don't use bomb if ship group is small.
      if self.num_ships <= 16:
        return 0

      if self.step < NEAR_ENDING_PHRASE_STEP and self.num_ships <= 25 :
        return 5

      # If having enough money and halite.
      if (estimate_halite(self.me) >
          estimate_halite(self.max_enemy) + self.c.spawn_cost * 2):
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
                       ShipTask.ATTACK_SHIPYARD)

      # One bomb at a time
      break

  def tight_defend_dist(self):
    # return 4 + max((self.num_ships - 15) // 5, 0)
    return TIGHT_ENEMY_SHIP_DEFEND_DIST

  def loose_defend_dist(self):
    return LOOSE_ENEMY_SHIP_DEFEND_DIST

  @property
  def shipyard_defend_dist(self):
    # return 3 if len(self.me.shipyard_ids) > 1 else 4
    return 4

  @property
  def home_grown_cell_dist(self):
    return self.tight_defend_dist()

  @property
  def is_beginning_phrase(self):
    return self.step <= BEGINNING_PHRASE_END_STEP

  def convert_to_shipyard(self):
    """Builds shipyard to maximize the total number of halite covered within
    |home_grown_cell_dist|."""
    MAX_SHIPYARD_NUM = 16
    MIN_NEXT_YARD_DIST = 6
    MAX_NEXT_YARD_DIST = 9
    HALITE_CELL_PER_SHIP = 2 if self.step < 60 else 3

    self.halite_ratio = -1
    # No ship left.
    if not self.num_ships:
      return

    def shipyard_num_by_ship_num():
      if self.num_ships >= 12:
        return min(2 + max((self.num_ships - 12) // 6, 0), MAX_SHIPYARD_NUM)
      return 1

    def shipyard_num_by_halite_ratio():
      num_halite_cells = 0
      for cell in self.halite_cells:
        min_dist, _ = self.get_nearest_home_yard(cell)
        if min_dist <= self.home_grown_cell_dist:
          num_halite_cells += 1

      num_yards = self.num_shipyards
      halite_ratio = num_halite_cells / (self.num_ships or 1)
      self.num_home_halite_cells = num_halite_cells
      self.halite_ratio = halite_ratio
      if halite_ratio < HALITE_CELL_PER_SHIP:
        num_yards += 1
        print('more ship: halite cell / ship =', halite_ratio)
      return num_yards

    def max_shipyard_num():
      return max(shipyard_num_by_ship_num(), shipyard_num_by_halite_ratio())

    # Reach max shipyard num.
    if self.num_shipyards >= max_shipyard_num():
      return

    def convert_threshold():
      threshold = MIN_HALITE_TO_BUILD_SHIPYARD

      # Use as much as I can.
      if (self.num_shipyards == 0 or
          self.board.step <= BEGINNING_PHRASE_END_STEP):
        threshold = self.c.convert_cost
      return max(self.c.convert_cost, threshold)

    def has_enough_halite(ship):
      return ship.halite + self.me.halite >= convert_threshold()

    def not_on_shipyard(ship):
      return ship.cell.shipyard_id is None

    def has_enemy_nearby(ship):
      return any(has_enemy_ship(cell, self.me)
                 for cell in get_neighbor_cells(ship.cell))

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
      shipyards = self.me.shipyards + [ship] # Fake ship as shipyard.
      for cell in self.halite_cells:
        if cell.position == ship.position:
          continue

        min_dist, _ = self.find_nearest(cell, shipyards)
        if min_dist <= self.home_grown_cell_dist:
          total_halite += cell.halite / min_dist # Use actual halite value.
          total_cell += 1
      return total_halite, total_cell

    ship_scores = [(score_ship_position(s), s)
                   for s in self.me.ships
                   if (has_enough_halite(s) and (not has_enemy_nearby(s))
                       and not_on_shipyard(s)
                       and within_predefined_range(s))]
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
      if (ship.task_type == ShipTask.STAY and
          ship.position == next_position):
        wt -= 10

      # If collecting halite
      if ((ship.task_type == ShipTask.GOTO_HALITE or
           ship.task_type == ShipTask.COLLECT) and
          target_cell.halite > 0):
        ship_to_poi = dist
        poi_to_yard, min_yard = self.get_nearest_home_yard(next_cell)
        if min_yard is None:
          poi_to_yard = 1
        expect_return = self.halite_per_turn(ship, target_cell, ship_to_poi,
                                             poi_to_yard)
        if expect_return < 0:
          return MIN_WEIGHT
        wt += expect_return

      # If go back home
      if ship.task_type == ShipTask.RETURN:
        wt += ship.halite / (dist + 1)

      # If goto enemy yard.
      if ship.task_type == ShipTask.ATTACK_SHIPYARD:
        # TODO: use what value as weight for destory enemy yard?
        wt += convert_cost / (dist + 1)

      # Do not step on shipyard
      if (ship.task_type != ShipTask.ATTACK_SHIPYARD and
          next_cell.shipyard_id and next_cell.shipyard.player_id != self.me.id):
        wt -= convert_cost / (dist + 1)

      if ship.task_type == ShipTask.ATTACK_SHIP:
        wt += convert_cost / (dist + 1)
        enemy = ship.target_enemy
        enemy_dist = manhattan_dist(next_position, enemy.position, self.c.size)
        wt += enemy.halite / (enemy_dist + 1)

      if ship.task_type == ShipTask.GUARD_SHIPYARD:
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
        if (ship.task_type == ShipTask.ATTACK_SHIPYARD and
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

    DO_NOT_STAY_TYPES = {ShipTask.GOTO_HALITE,
                         ShipTask.RETURN,
                         ShipTask.ATTACK_SHIPYARD}
    g = nx.Graph()
    for ship in ships:
      random.shuffle(POSSIBLE_MOVES)
      for move in POSSIBLE_MOVES:
        # Skip stay on home cells for some task type.
        if ship.task_type in DO_NOT_STAY_TYPES and move == Point(0, 0):
          continue

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
    SHIP_NUM_HARD_LIMIT = 100

    # When leading, convert as much as possible.
    def max_ship_num():
      by_cash = max(0, (self.me_halite - 3000) // 1000) + MAX_SHIP_NUM

      by_enemy_halite = 0
      if self.me.total_halite > self.max_enemy_halite + 6 * self.c.spawn_cost:
        by_enemy_halite = SHIP_NUM_HARD_LIMIT

      return min(SHIP_NUM_HARD_LIMIT, max(by_cash, by_enemy_halite))

    def spawn(yard):
      self.cost_halite += self.c.spawn_cost
      yard.next_action = ShipyardAction.SPAWN

    def spawn_threshold():
      if self.step <= BEGINNING_PHRASE_END_STEP:
        return self.c.spawn_cost
      return MIN_HALITE_TO_BUILD_SHIP

    # Too many ships.
    mx = max_ship_num()
    if self.num_ships >= max_ship_num():
      return

    # No more ships after ending.
    if self.num_ships >= 3 and self.step >= 320:
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
                         ShipTask.RETURN)

  initial_shipyard_set = False
  initial_yard_position = None
  initial_ship_position = None

  def convert_first_shipyard(self):
    """Strategy for convert the first shipyard."""
    assert self.num_ships == 1, self.num_ships

    ship = self.me.ships[0]
    if not self.initial_ship_position:
      ShipStrategy.initial_ship_position = ship.position

    def expected_coverted_halite(candidate_cell):
      expected_halite = 0
      current_halite = 0
      num_halite_cells = 0
      for cell in self.halite_cells:
        # shipyard will destory the halite under it.
        if candidate_cell.position == cell.position:
          continue

        dist = manhattan_dist(cell.position, candidate_cell.position,
                              self.c.size)
        # TODO(wangfei): try larger value?
        if dist <= self.home_grown_cell_dist and cell.halite > 0:
          expected_halite += self.halite_per_turn(None, cell, dist, dist)
          current_halite += cell.halite
          num_halite_cells += 1

      # candidate_cell.position)
      print(candidate_cell.position, "expexted=", expected_halite, 'current=',
            current_halite, 'n=', num_halite_cells)
      return (expected_halite, current_halite, dist)

    def get_coord_range(v):
      DELTA = 1
      MARGIN = 4
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

    self.assign_task(ship, self.board[self.initial_yard_position],
                      ShipTask.INITIAL_SHIPYARD)
    if ship.position == self.initial_yard_position:
      ship.next_action = ShipAction.CONVERT
      ShipStrategy.initial_shipyard_set = True

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

    def print_player(player, end='\n'):
      num_ships = len(player.ship_ids)
      num_shipyards = len(player.shipyard_ids)
      print('  p[%s](s=%s, y=%s, h=%s, c=%s, mc=%s)' % (player.id,
                                                   num_ships,
                                                   num_shipyards,
                                                   player.halite,
                                                   cargo(player),
                                                   mean_cargo(player)), end=end)

    def print_ship_task_type():
      task = Counter()
      for ship in self.me.ships:
        task[ship.task_type] += 1
      items = sorted(task.items(), key=lambda x: x[0].name)
      print(", ".join("%s=%s(%.0f%%)" % (k.name, v, v/self.num_ships*100) for k, v in items))

    print('#%s' % self.step, 'halite(mean=%s, std=%s)' % (int(self.mean_halite_value),
                                                          int(self.std_halite_value)),
          'hh=(n=%s, m=%s, n/s=%.1f)' % (
            self.num_home_halite_cells,
            int(self.mean_home_halite),
            self.halite_ratio))
    print_player(self.me, end=' ')

    enemy = sorted(self.board.opponents, key=lambda x: -(len(x.ship_ids)))[0]
    print_player(enemy, end='\n')
    print_ship_task_type()
    print()


  def halite_per_turn(self,
                      ship,
                      poi: Cell,
                      ship_to_poi,
                      poi_to_yard,
                      min_mine=1):
    """Computes the expected return for mining with optimial steps.

    TODO(wangfei): we could use small panelty for return home dist
    to mimic the we don't want back home.
    """
    carry = ship.halite if ship else 0
    halite = poi.halite
    if ship and has_enemy_ship(poi, self.me):
      # Dist to move to neighour cell of POI.
      dist = max(0, ship_to_poi - 1)

      # Halite will decrease if there is ship sitting on it.
      halite_left = poi.halite * HALITE_RETENSION_BY_DIST[dist]

      # Give up if my ship has more halite then enemy.
      enemy_halite = poi.ship.halite + (poi.halite - halite_left)
      if ship and ship.halite >= enemy_halite:
        return -1000

      halite = poi.halite + poi.ship.halite

    # By dividing 2, it means collecting multiple cell to share return home cost
    # travel = ship_to_poi + (poi_to_yard / 2.0)
    travel = ship_to_poi + poi_to_yard
    opt_steps = optimal_mining_steps(carry, halite, travel)
    if opt_steps < min_mine:
      opt_steps = min_mine
    total_halite = carry + (1 - HALITE_RETENSION_BY_DIST[opt_steps]) * halite
    return total_halite / (travel + opt_steps)

  def get_trapped_enemy_ships(self, max_attack_num):
    """A enemy is trapped if there're at least one ship in each quadrant."""
    # Do not attack enemy during ending.
    if self.step >= NEAR_ENDING_PHRASE_STEP:
      return

    is_strong  = int(self.num_ships >= 23)
    MAX_ATTACK_DIST = 4 + is_strong
    MIN_ATTACK_QUADRANT_NUM = 3 - is_strong

    def get_attack_ships(enemy):
      for ship in self.my_idle_ships:
        dist = manhattan_dist(ship.position, enemy.position, self.c.size)
        if dist <= MAX_ATTACK_DIST and ship.halite < enemy.halite:
          yield dist, ship

    def annotate_by_quadrant(dist_ships, enemy):
      """Sort to make sure at least one ship is selected in each quadrant."""
      quadrants = set()
      for (dist, ship) in dist_ships:
        q = get_quadrant(ship.position - enemy.position)
        q_exist = int(q in quadrants)
        quadrants.add(q)
        yield (q_exist, dist), ship

    for enemy in self.enemy_ships:
      dist_ships = get_attack_ships(enemy)
      dist_ships = list(annotate_by_quadrant(dist_ships, enemy))
      dist_ships.sort(key=lambda x: x[0])
      quadrant_num = len({
          get_quadrant(ship.position - enemy.position)
          for _, ship in dist_ships
      })
      if quadrant_num >= MIN_ATTACK_QUADRANT_NUM:
        yield enemy, [ship for _, ship in dist_ships][:max_attack_num]

  def optimal_assigntment(self):
    ATTACK_PER_ENEMY = 5
    SHIPYARD_DUPLICATE_NUM = 5

    def shipyard_duplicate_num():
      if self.step >= NEAR_ENDING_PHRASE_STEP:
        return 1
      return SHIPYARD_DUPLICATE_NUM

    ships = list(self.my_idle_ships)
    halites = [c for c in self.halite_cells if c.halite >= c.keep_halite_value]

    # Shipyards is duplicated to allow multiple ships having a same target.
    shipyards = [y.cell for y in self.me.shipyards] * shipyard_duplicate_num()


    # Attack enemy.
    trapped_enemy_ships = list(self.get_trapped_enemy_ships(ATTACK_PER_ENEMY))
    enemy_cells = [e.cell for e, _ in trapped_enemy_ships] * ATTACK_PER_ENEMY
    attack_pairs = {(s.id, e.id)
                    for e, ships in trapped_enemy_ships
                    for s in ships}

    # Guard shipyard.
    offended_shipyards = list(self.get_offended_shipyard())
    offended_cells = [y.cell for y, _ in offended_shipyards]
    guard_paris = {(s.id, y.id)
                   for y, ships in offended_shipyards
                   for s in ships}

    pois = halites + shipyards + enemy_cells + offended_cells

    def is_halite_column(x):
      return x < len(halites)

    def is_shipyard_column(x):
      return len(halites) <= x < len(halites) + len(shipyards)

    def is_enemy_column(x):
      return (len(halites) + len(shipyards) <= x
              and x < len(halites) + len(shipyards) + len(enemy_cells))

    # Value matrix for ship target assginment
    # * row: ships
    # * column: halite cells + shipyards with duplicates.
    # TODO(wangfei): can we add enemy to this matrix?
    C = np.zeros((len(ships), len(pois)))
    for i, ship in enumerate(ships):
      for j, poi in enumerate(pois):
        # Init distances: from ship to POI and POI to the nearest yard.
        ship_to_poi = manhattan_dist(ship.position, poi.position, self.c.size)
        poi_to_yard, min_yard = self.get_nearest_home_yard(poi)
        if min_yard is None:
          poi_to_yard = 1

        if is_halite_column(j):
          # If the target is a halite cell, with enemy considered.
          v = self.halite_per_turn(ship, poi, ship_to_poi, poi_to_yard)
        elif is_shipyard_column(j):
          # If the target is a shipyard.
          if ship_to_poi > 0:
            v = ship.halite / ship_to_poi
          else:
            # The ship is on a shipyard.
            v = 0
        elif is_enemy_column(j):
          # If attack enemy
          enemy = poi.ship
          v = -99999  # not exists edge.
          if (ship.id, enemy.id) in attack_pairs:
            # Discount by 0.5
            v = (self.c.spawn_cost + enemy.halite * 0.5) / ship_to_poi
        else:
          # If shipyard is offended.
          yard = poi.shipyard
          v = -99999
          if (ship.id, yard.id) in guard_paris:
            v = (self.c.spawn_cost + self.c.convert_cost) / (ship_to_poi or 1)

        C[i, j] = v

    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)
    # assert len(rows) == len(ships), "ships=%s, halites=%s" % (len(ships),
    # len(halites))
    for ship_idx, poi_idx in zip(rows, cols):
      ship = ships[ship_idx]
      poi_cell = pois[poi_idx]
      # print('send ship(id=%s, p=%s, h=%s)' % (ship.id, ship.position,
      # ship.halite),
      # 'to poi_cell(p=%s, h=%s)' % (poi_cell.position,
      # poi_cell.halite))
      enemy = None
      if is_halite_column(poi_idx):
        if ship.position == poi_cell.position:
          task_type = ShipTask.COLLECT
        else:
          task_type = ShipTask.GOTO_HALITE
      elif is_shipyard_column(poi_idx):
        task_type = ShipTask.RETURN
      elif is_enemy_column(poi_idx):
        task_type = ShipTask.ATTACK_SHIP
        enemy = poi_cell.ship
      else:
        task_type = ShipTask.GUARD_SHIPYARD
        enemy = poi_cell.shipyard.offend_enemy
        print('guide task: ', ship.position, poi_cell.position,
              poi_cell.shipyard.offend_enemy.position)

      self.assign_task(ship, poi_cell, task_type, enemy=enemy)

  def manhattan_dist(self, p, q):
    return manhattan_dist(p.position, q.position, self.c.size)

  def get_offended_shipyard(self):

    def offend_enemy_ships(yard):
      not_enough_halite_to_spawn = self.me_halite < self.c.spawn_cost
      for enemy in self.enemy_ships:
        if self.manhattan_dist(enemy, yard) > self.shipyard_defend_dist:
          continue

        # If the enemy has money, then I'll just let me send it for me.
        if not_enough_halite_to_spawn or enemy.halite == 0:
          yield enemy

    def get_defend_ships(yard, enemy, enemy_to_yard_dist):
      for ship in self.my_idle_ships:
        dist_to_yard = self.manhattan_dist(ship, yard)
        if (dist_to_yard < enemy_to_yard_dist
            or (dist_to_yard == enemy_to_yard_dist
                and (enemy.halite > 0 and ship.halite < enemy.halite
                     or enemy.halite == 0 and ship.halite == 0))):
          yield ship

    for yard in self.me.shipyards:
      min_enemy_dist, enemy = self.find_nearest(yard.cell,
                                                offend_enemy_ships(yard))
      if enemy is None:
        continue
      # No need guard shipyard if enemy has halite (by turn order, spawn comes
      # before collision)
      if yard.next_action == ShipyardAction.SPAWN and enemy.halite > 0:
        continue

      defend_ships = list(get_defend_ships(yard, enemy, min_enemy_dist))
      if defend_ships:
        yard.offend_enemy = enemy
        yield yard, defend_ships

  def execute(self):
    self.collect_game_info()

    if self.initial_shipyard_set:
      self.convert_to_shipyard()
      self.spawn_ships()

      self.bomb_enemy_shipyard()

      self.final_stage_back_to_shipyard()
      self.optimal_assigntment()
    else:
      self.convert_first_shipyard()

    self.compute_ship_moves()
    self.spawn_if_shipyard_in_danger()
    if not self.simulation:
      self.print_info()


def init_globals(board):
  growth_factor = board.configuration.regen_rate + 1.0
  retension_rate_rate = 1.0 - board.configuration.collect_rate
  size = board.configuration.size

  # init constants
  global HALITE_GROWTH_BY_DIST
  if not HALITE_GROWTH_BY_DIST:
    HALITE_GROWTH_BY_DIST = [growth_factor**d for d in range(size**2 + 1)]

  global HALITE_RETENSION_BY_DIST
  if not HALITE_RETENSION_BY_DIST:
    HALITE_RETENSION_BY_DIST = [
        retension_rate_rate**d for d in range(size**2 + 1)
    ]
  # print(HALITE_GROWTH_BY_DIST, HALITE_RETENSION_BY_DIST)


@board_agent
def agent(board):
  init_globals(board)
  ShipStrategy(board).execute()
