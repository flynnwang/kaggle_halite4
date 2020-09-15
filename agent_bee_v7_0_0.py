# -*- coding: utf-8 -*-
"""
v7_0_0 <- v4_16_10

* Discount 0.7, min halite 50
* Always enable enemy_carry, poi_to_yard / 3
* fine tune keep_halite_value, start grow if shiypard >= 4
* shipyard convert maximize toward convered cells.
* Super strike, grow dist by (s - 23) // 6
"""

import sys
import random
import timeit
import logging
from collections import Counter, defaultdict
from enum import Enum, auto

import numpy as np
import scipy.optimize
from kaggle_environments.envs.halite.helpers import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# Mute print.
def print(*args, **kwargs):
  pass


MIN_WEIGHT = -99999

BEGINNING_PHRASE_END_STEP = 60
CLOSING_PHRASE_STEP = 300
NEAR_ENDING_PHRASE_STEP = 350

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

# Controls the number of ships.
MAX_SHIP_CARGO = 500

# Threshold for attack enemy nearby my shipyard
TIGHT_ENEMY_SHIP_DEFEND_DIST = 5
LOOSE_ENEMY_SHIP_DEFEND_DIST = 7
AVOID_COLLIDE_RATIO = 0.95

SHIPYARD_TIGHT_COVER_DIST = 2
SHIPYARD_LOOSE_COVER_DIST = 6
MIN_BOMB_ENEMY_SHIPYARD_DIST = 4

ALLEY_SUPPORT_DIST = 5
MAX_SUPPORT_NUM = 2

MIN_ENEMY_TO_RUN = 3

STRIKE_COOLDOWN_STEPS = 10

SUPER_STRIKE_COOLDOWN = 40
SUPER_STRIKE_ATTACK_MIN_DIST = 6 # use large value...
SUPER_MIN_STRIKE_SHIP_NUM = 23
SUPER_STRIKE_MIN_NO_WORK_SHIP_NUM = 10
SUPER_STRIKE_HALITE_GAIN = 700

# For building shipyard after a successful strike
STRIKE_CALL_FOR_SHIPYARD_STEPS = 25

MIN_COVER_RATIO = 0.6
# Threshod used to send bomb to enemy shipyard
POSSIBLE_MOVES = [
    Point(0, 0),
    Point(0, 1),
    Point(0, -1),
    Point(1, 0),
    Point(-1, 0)
]

TURNS_OPTIMAL = np.array(
    [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
     [0, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7],
     [0, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7],
     [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7],
     [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7],
     [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7],
     [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7],
     [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7],
     [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
     [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
     [0, 0, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7],
     [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7],
     [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
     [0, 0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
     [0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
     [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
     [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
     [0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6],
     [0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
     [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6],
     [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
     [0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5],
     [0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
     [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5],
     [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
     [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5],
     [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
     [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

# cached values
HALITE_RETENSION_BY_DIST = []
HALITE_GROWTH_BY_DIST = []
HALITE_COLLECT_STEPS = []
MANHATTAN_DISTS = None


def get_quadrant(p: Point):
  if p.x > 0 and p.y >= 0:
    return 1
  if p.x <= 0 and p.y > 0:
    return 2
  if p.x < 0 and p.y <= 0:
    return 3
  if p.x >= 0 and p.y < 0:
    return 4
  # assert p == Point(0, 0), "not exist quadrant: %s %s" % (p.x, p.y)
  return 0


def optimal_mining_steps(C, H, rt_travel):
  # How many turns should we plan on mining?
  # C=carried halite, H=halite in square, rt_travel=steps to square and back to shipyard
  if C == 0:
    ch = 0
  elif H == 0:
    ch = TURNS_OPTIMAL.shape[0] - 1  # ?
  else:
    # ch = int(np.log(C / H) * 2.5 + 5.5)
    ch = int(np.log(C / H) * 10 + 24)
    ch = min(max(ch, 0), TURNS_OPTIMAL.shape[0] - 1)
  # rt_travel = int(np.clip(rt_travel, 0, TURNS_OPTIMAL.shape[1] - 1))
  rt_travel = int(min(max(rt_travel, 0), TURNS_OPTIMAL.shape[1] - 1))
  return TURNS_OPTIMAL[ch, rt_travel]


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


def axis_manhattan_dists(a: Point, b: Point, size):

  def dist(x, y):
    v = abs(x - y)
    return min(v, size - v)

  return dist(a.x, b.x), dist(a.y, b.y)


def manhattan_dist(a: Point, b: Point, size):
  if MANHATTAN_DISTS:
    return MANHATTAN_DISTS[a.x * size + a.y][b.x * size + b.y]

  dist_x, dist_y = axis_manhattan_dists(a, b, size)
  return dist_x + dist_y


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

  # assert False, '%s, %s' % (position, next_position)
  return None


def make_move(position, move, board_size):
  return (position + move) % board_size


def get_neighbor_cells(cell, include_self=False):
  neighbor_cells = [cell] if include_self else []
  neighbor_cells.extend([cell.north, cell.south, cell.east, cell.west])
  return neighbor_cells


def collect_times(h, stop_h=50, collect_rate=0.25):
  if stop_h < 30:
    return 0
  t = 0
  while h >= stop_h and h > 0:
    h = h - int(h * collect_rate)
    t += 1
  return t

def init_globals(board):
  growth_factor = board.configuration.regen_rate + 1.0
  retension_rate_rate = 1.0 - board.configuration.collect_rate
  size = board.configuration.size

  global HALITE_GROWTH_BY_DIST
  if not HALITE_GROWTH_BY_DIST:
    HALITE_GROWTH_BY_DIST = [growth_factor**d for d in range(size**2 + 1)]

  global HALITE_RETENSION_BY_DIST
  if not HALITE_RETENSION_BY_DIST:
    HALITE_RETENSION_BY_DIST = [
        retension_rate_rate**d for d in range(size**2 + 1)
    ]

  global MANHATTAN_DISTS
  dists = np.zeros((size**2, size**2), dtype=int)
  with Timer("Init manhattan_dist"):
    for c1 in board.cells.values():
      for c2 in board.cells.values():
        a = c1.position
        b = c2.position
        d = manhattan_dist(a, b, size)
        dists[a.x * size + a.y][b.x * size + b.y] = d
  MANHATTAN_DISTS = dists.tolist()

  global HALITE_COLLECT_STEPS
  HALITE_COLLECT_STEPS = {}

  with Timer("Init HALITE_COLLECT_STEPS"):
    for h in range(0, 501):
      for stop in range(0, 501):
        HALITE_COLLECT_STEPS[(h, stop)] = collect_times(h, stop)


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


class StrategyBase:
  """Class with board related method."""

  @property
  def me(self):
    return self.board.current_player

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
  def sz(self):
    return self.c.size

  @property
  def num_shipyards(self):
    return len(self.me.shipyard_ids)

  @property
  def tight_defend_dist(self):
    # return 4 + max((self.num_ships - 15) // 5, 0)
    return TIGHT_ENEMY_SHIP_DEFEND_DIST

  @property
  def loose_defend_dist(self):
    return LOOSE_ENEMY_SHIP_DEFEND_DIST

  @property
  def home_grown_cell_dist(self):
    return self.tight_defend_dist

  @property
  def is_beginning_phrase(self):
    return self.step <= BEGINNING_PHRASE_END_STEP

  @property
  def is_closing_phrase(self):
    return self.step >= CLOSING_PHRASE_STEP

  @property
  def is_final_phrase(self):
    return self.step >= NEAR_ENDING_PHRASE_STEP

  @property
  def my_idle_ships(self):
    """All ships without task assignment."""
    for ship in self.ships:
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

  def manhattan_dist(self, p, q):
    return manhattan_dist(p.position, q.position, self.c.size)

  def nearest_shipyards(self, cell: Cell, shipyards):
    dist_yards = [(self.manhattan_dist(y, cell), y) for y in shipyards]
    dist_yards = sorted(dist_yards, key=lambda x: x[0])
    return dist_yards

  def find_nearest_enemy(self, cell: Cell, enemy_ships):
    """Nearest enemy with least halite."""
    if not isinstance(enemy_ships, list):
      enemy_ships = list(enemy_ships)

    enemy_ships = sorted(enemy_ships,
                         key=lambda s: (self.manhattan_dist(cell, s), s.halite))
    for enemy in enemy_ships:
      return self.manhattan_dist(cell, enemy), enemy
    return 9999, None

  def get_nearest_home_yard(self, cell):
    if not hasattr(cell, 'home_yard_info'):
      cell.nearest_home_yards = self.nearest_shipyards(cell, self.shipyards)
      cell.home_yard_info = (9999, None)
      if cell.nearest_home_yards:
        cell.home_yard_info = cell.nearest_home_yards[0]
    return cell.home_yard_info

  def get_nearest_enemy_yard(self, cell):
    if not hasattr(cell, 'enemy_yard_info'):
      cell.nearest_enemy_yards = self.nearest_shipyards(cell,
                                                        self.enemy_shipyards)
      cell.enemy_yard_info = (9999, None)
      if cell.nearest_enemy_yards:
        cell.enemy_yard_info = cell.nearest_enemy_yards[0]
    return cell.enemy_yard_info

  def has_enemy_nearby(self, cell):
    return any(
        has_enemy_ship(c, self.me)
        for c in get_neighbor_cells(cell, include_self=True))

  def update(self, board):
    self.board = board

    # Cache it to eliminate repeated list constructor.
    self.shipyards = self.me.shipyards
    self.ships = self.me.ships

  def execute(self):
    pass

  def __call__(self):
    self.execute()


class FollowerDetector(StrategyBase):

  # >= 2 is considered as following.
  FOLLOW_COUNT = 2

  def __init__(self):
    self.board = None
    self.followers = defaultdict(Counter)  # ship_id => {follower_id: count}

  def add(self, ship_id, followers):
    """Note: follower.halite < ship.halite"""
    latest_follower_ids = {f.id for f in followers}

    # Cleanup non-followers
    follow_count = self.followers[ship_id]
    for f in list(follow_count.keys()):
      if f not in latest_follower_ids:
        del follow_count[f]

    for f in latest_follower_ids:
      follow_count[f] += 1

  def update(self, board):
    """Updates follow info with the latest board state."""
    super().update(board)

    for ship in self.ships:
      enemies = []
      for nb_cell in get_neighbor_cells(ship.cell):
        # If no ship there
        if nb_cell.ship_id is None:
          continue

        # Not enemy ship
        enemy = nb_cell.ship
        if enemy.player_id == self.me.id:
          continue

        # Not a threat
        if enemy.halite >= ship.halite:
          continue
        enemies.append(enemy)

      self.add(ship.id, enemies)

    self.enemy_ship_index = {s.id: s for s in self.enemy_ships}

  def is_followed(self, ship: Ship):
    """Returns true if the ship of mine is traced by enemy."""
    follower_count = self.followers.get(ship.id)
    # if len(follower_count) >= 2:
    # return True
    for fc in follower_count.values():
      if fc >= self.FOLLOW_COUNT:
        return True
    return False

  def get_followers(self, ship: Ship):
    follower_count = self.followers[ship.id]
    return [self.enemy_ship_index[sid] for sid in follower_count]


class InitializeFirstShipyard(StrategyBase):

  def __init__(self):
    super().__init__()
    self.first_shipyard_set = False
    self.initial_yard_position = None
    self.initial_ship_position = None

  def estimate_cell_halite(self, candidate_cell):
    expected_halite = 0
    current_halite = 0
    num_halite_cells = 0
    for cell in self.halite_cells:
      # shipyard will destory the halite under it.
      if candidate_cell.position == cell.position:
        continue

      dist = self.manhattan_dist(cell, candidate_cell)
      # TODO(wangfei): try larger value?
      if dist <= self.home_grown_cell_dist and cell.halite > 0:
        expected_halite += self.halite_per_turn(None, cell, dist, dist)
        current_halite += cell.halite
        num_halite_cells += 1
    return expected_halite, current_halite, dist

  def select_initial_cell(self):

    def get_coord_range(v):
      DELTA = 0
      MARGIN = 5
      if v == 5:
        v_min, v_max = MARGIN, 5 + DELTA
      else:
        v_min, v_max = 15 - DELTA, 20 - MARGIN
      return v_min, v_max

    position = self.initial_ship_position
    x_min, x_max = get_coord_range(position.x)
    y_min, y_max = get_coord_range(position.y)
    for cell in self.board.cells.values():
      position = cell.position
      if (x_min <= position.x <= x_max and y_min <= position.y <= y_max):
        yield self.estimate_cell_halite(cell), cell

  def convert_first_shipyard(self):
    """Strategy for convert the first shipyard."""
    assert self.num_ships == 1, self.num_ships

    ship = self.ships[0]
    if not self.initial_ship_position:
      self.initial_ship_position = ship.position

      candidate_cells = list(self.select_initial_cell())
      if candidate_cells:
        candidate_cells.sort(key=lambda x: x[0], reverse=True)
        value, yard_cell = candidate_cells[0]
        self.initial_yard_position = yard_cell.position
        print(
            "Ship initial:", self.initial_ship_position, 'dist=',
            manhattan_dist(self.initial_ship_position,
                           self.initial_yard_position, self.c.size),
            'selected yard position:', self.initial_yard_position, 'value=',
            value)

    self.assign_task(ship, self.board[self.initial_yard_position],
                     ShipTask.INITIAL_SHIPYARD)
    if ship.position == self.initial_yard_position:
      ship.next_action = ShipAction.CONVERT
      self.first_shipyard_set = True


class GradientMap(StrategyBase):

  def __init__(self):
    self.board = None
    self.enemy_gradient = None
    self.nearby_positions_cache = {}
    self.nearby_enemies = {}  # position => enemy list, clear up every step.

  def update(self, board):
    super().update(board)
    self.nearby_enemies.clear()

  def _get_nearby_positions(self, center: Cell, max_dist):
    visited = set()
    nearby_positions = []

    def dfs(c: Cell):
      if c.position in visited:
        return
      visited.add(c.position)

      if self.manhattan_dist(c, center) > max_dist:
        return
      nearby_positions.append(c.position)
      for next_cell in get_neighbor_cells(c):
        dfs(next_cell)

    dfs(center)
    return nearby_positions

  def get_nearby_cells(self, center: Cell, max_dist):
    key = (center.position, max_dist)
    if key in self.nearby_positions_cache:
      positions = self.nearby_positions_cache[key]
    else:
      positions = self._get_nearby_positions(center, max_dist)
      self.nearby_positions_cache[key] = positions
    return (self.board[p] for p in positions)

  def compute_gradient(self, center_cells, max_dist, value_func):
    gradient = np.zeros((self.sz, self.sz))
    for center in center_cells:
      for nb_cell in self.get_nearby_cells(center, max_dist):
        p = nb_cell.position
        gradient[p.x, p.y] += value_func(center, nb_cell)
    return gradient

  def get_enemy_gradient(self,
                         center_cell,
                         max_dist=2,
                         broadcast_dist=1,
                         halite=999999,
                         normalize=False):
    """The amount enemy can hurt me."""

    def nearby_enemy_cells():
      for cell in self.get_nearby_cells(center_cell, max_dist):
        if has_enemy_ship(cell, self.me):
          yield cell

    def enemy_cost(dist):
      if dist > broadcast_dist:
        return 0
      return self.c.spawn_cost / (dist or 1)

    def enemy_value(enemy_cell, nb_cell):
      enemy = enemy_cell.ship
      dist = self.manhattan_dist(nb_cell, enemy_cell)
      if enemy.halite < halite:
        return enemy_cost(dist)
      return 0

    g = self.compute_gradient(nearby_enemy_cells(), max_dist, enemy_value)
    if normalize:
      g = g / (np.max(g) + 0.1)
    return g

  def count_nearby_true_enemy(self, halite_cell, ship, with_cell_halite=True):
    NEARBY_ENEMY_DIST = 3

    def get_nearby_enemies():
      if halite_cell.position in self.nearby_enemies:
        return self.nearby_enemies[halite_cell.position]

      nearby_cells = self.get_nearby_cells(halite_cell, NEARBY_ENEMY_DIST)
      enemies = [c.ship for c in nearby_cells if has_enemy_ship(c, self.me)]
      self.nearby_enemies[halite_cell.position] = enemies
      return enemies

    next_step_halite = ship.halite
    if with_cell_halite:
      next_step_halite += self.c.collect_rate * halite_cell.halite
    return sum(1 for enemy in get_nearby_enemies()
               if enemy.halite < next_step_halite)

  def get_halite_gradient_map(self,
                              max_dist=3,
                              min_halite=0,
                              include_center=True,
                              normalize=True):

    def all_halite_cells():
      for cell in self.board.cells.values():
        if cell.halite > min_halite:
          yield cell

    def halite_value(cell, nb_cell):
      dist = self.manhattan_dist(nb_cell, cell)
      # Do not account for the halite of the cell itself.
      if dist == 0 and not include_center:
        return 0
      return cell.halite / (dist + 1)

    g = self.compute_gradient(all_halite_cells(), max_dist, halite_value)
    if normalize:
      g = g / (np.max(g) + 0.1)
    return g


class ShipStrategy(InitializeFirstShipyard, StrategyBase):
  """Sends every ships to the nearest cell with halite.

  cell:
    |is_targetd|: prevent more than 1 alley ships choose the same halite.

  ship:
    |next_cell|: next cell location of the ship.
    |has_assignment|: has already has task assignment for this ship.
    |target_cell|: send ship to this cell, may be the same of current cell.
    |task_type|: used to rank ship for moves.
  """

  def __init__(self, simulation=False):
    super().__init__()
    self.board = None
    self.simulation = simulation
    self.follower_detector = FollowerDetector()
    self.gradient_map = GradientMap()
    self.history_shipyard_positions = set()
    self.strike_shipyard = None
    self.strike_success_position = None
    self.strike_success_step = 0
    self.num_covered_halite_cells = 0
    self.strike_ship_num = 0
    self.call_for_shipyard = False

  def update(self, board):
    """Updates board state at each step."""
    if self.board is None:
      init_globals(board)

    super().update(board)

    self.board = board
    self.cost_halite = 0
    self.halite_ratio = -1
    self.num_home_halite_cells = 0
    self.mean_home_halite = 100

    self.init_halite_cells()

    # Default ship to stay on the same cell without assignment.
    for ship in self.ships:
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.task_type = ShipTask.STAY

    self.follower_detector.update(board)
    self.gradient_map.update(board)
    self.halite_gradient = self.gradient_map.get_halite_gradient_map(max_dist=3)

    # Save shipyard positions.
    for yard in self.shipyards:
      self.history_shipyard_positions.add(yard.position)

    # Update strike info if the shipyard is no longer there
    if (self.strike_shipyard and
        self.board[self.strike_shipyard.position].shipyard_id !=
        self.strike_shipyard.id):
      self.strike_success_position = self.strike_shipyard.position
      self.strike_success_step = self.step - 1
      self.strike_shipyard = None
      self.call_for_shipyard = True

    # Note: call for ship will spawn shipyard, may not at the expeted position
    # Effect: bonus shipyard every time we make a success strike!
    if self.step - self.strike_success_step > STRIKE_CALL_FOR_SHIPYARD_STEPS:
      self.call_for_shipyard = False

    # if self.strike_shipyard:
    # logger.info(f"  strike_shipyard.position={self.strike_shipyard.position},"
    # f" yard_id={self.strike_shipyard.id} ,")
    # logger.info(f"STEP={self.step}, strike_success_position={self.strike_success_position},"
    # f" strike_success_step={self.strike_success_step}")

  @property
  def halite_cover_ratio(self):
    return self.num_covered_halite_cells / len(self.halite_cells)

  def init_halite_cells(self):
    HOME_GROWN_CELL_MIN_HALITE = 80
    MAX_STEP_FACTOR = 2

    max_enemy_ship_num = max(len(p.ship_ids) for p in self.board.opponents)
    self.ship_to_enemy_ratio = self.num_ships / (max_enemy_ship_num + 0.1)

    def home_extend_dist():
      return max(self.num_ships // 10, 2)

    def is_home_grown_cell(cell):
      num_covered = len(cell.covering_shipyards)
      return (num_covered >= 2 or num_covered > 0 and
              cell.covering_shipyards[0][0] <= home_extend_dist())

    def keep_halite_value(cell):
      discount_factor = 0.7
      if self.step < 30:
        discount_factor = 0.8

      board_halite_value = self.mean_halite_value * discount_factor

      # Collect larger ones first
      threshold = board_halite_value

      if self.is_final_phrase:
        return threshold

      if is_home_grown_cell(cell):
        home_halite_value = board_halite_value * max(1.0, self.ship_to_enemy_ratio)

        if (self.halite_cover_ratio > MIN_COVER_RATIO or
            self.ship_to_enemy_ratio > 1.1 or
            self.num_shipyards >= 4 or
            self.step >= 260):
          home_halite_value = self.mean_halite_value * self.ship_to_enemy_ratio
          F = self.num_ships / 10 + 1
          home_halite_value = max(home_halite_value,
                                  F * HOME_GROWN_CELL_MIN_HALITE)

        START_GROW_YARD_NUM = 3
        # Grow by shipyard
        grow_by_shipyard = max(0, self.num_shipyards - START_GROW_YARD_NUM) * 30
        home_halite_value += grow_by_shipyard

        if self.num_shipyards > START_GROW_YARD_NUM:
          if len(cell.covering_shipyards) == 2:
            home_halite_value += 10 * max(0, self.num_shipyards - START_GROW_YARD_NUM)
          if len(cell.covering_shipyards) == 3:
            home_halite_value += 20 * max(0, self.num_shipyards - START_GROW_YARD_NUM)


        # Harvest for more ships.
        # if 240 <= self.step <= 265 and self.num_ships >= 23:
          # home_halite_value = self.mean_halite_value

        threshold = max(home_halite_value, threshold)

      return min(threshold, 480)

    # Init halite cells
    self.halite_cells = []
    for cell in self.board.cells.values():
      cell.is_targetd = False
      if cell.halite > 0:
        self.halite_cells.append(cell)

    # Initialize covered cells by shipyards.
    for cell in self.halite_cells:
      # Populate cache
      self.get_nearest_home_yard(cell)
      home_yards = [
          x for x in cell.nearest_home_yards
          if x[0] <= self.home_grown_cell_dist
      ]
      cell.covering_shipyards = home_yards

    self.mean_halite_value = 0
    if self.halite_cells:
      halite_values = [c.halite for c in self.halite_cells]
      self.mean_halite_value = np.mean(halite_values)

    self.total_collect_steps = 0
    self.total_collectable_halite = 0

    self.home_halite_cell_num = 0
    self.below_keep_halite_cell_num = 0
    for cell in self.halite_cells:
      cell.keep_halite_value = keep_halite_value(cell)
      if len(cell.covering_shipyards) > 0:
        lower_bound = 50
        self.total_collectable_halite += max(cell.halite - lower_bound, 0)

        h = min(500, max(0, int(cell.halite)))
        self.total_collect_steps += HALITE_COLLECT_STEPS[(h, lower_bound)]

      if is_home_grown_cell(cell):
        self.home_halite_cell_num += 1
        if cell.halite < cell.keep_halite_value:
          self.below_keep_halite_cell_num += 1

    self.home_empty_halite_cell_ratio = self.below_keep_halite_cell_num / (self.home_halite_cell_num + 0.1)

    # path = '/home/wangfei/data/20200801_halite/output/home_empty_halite_cell_ratio.csv'
    # with open(path, 'a') as f:
      # f.write(f'{self.num_ships},{self.home_halite_cell_num},{self.below_keep_halite_cell_num}\n')


  @property
  def me_halite(self):
    return self.me.halite - self.cost_halite

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
    self.num_covered_halite_cells = len(home_cells)

    # Player info
    self.me.total_halite = self.me.halite + cargo(self.me)

    self.max_enemy_halite = -1
    self.max_enemy = None
    self.total_enemy_ship_num = 0
    for p in self.board.opponents:
      p.total_halite = p.halite + cargo(p)
      if p.total_halite >= self.max_enemy_halite:
        self.max_enemy_halite = p.halite
        self.max_enemy = p

      self.total_enemy_ship_num += len(p.ship_ids)

  def bomb_enemy_shipyard(self):
    """Having enough farmers, let's send ghost to enemy shipyard."""

    def is_enemy_weak(enemy_yard, factor=2):
      return enemy_yard.player.halite < self.c.spawn_cost * factor

    def is_enemy_yard_within_home_boundary(enemy_yard, enemy_yard_dist):
      if enemy_yard_dist <= 2:
        return True
      self.get_nearest_home_yard(enemy_yard.cell)
      home_yards = [
          x for x in enemy_yard.cell.nearest_home_yards
          if x[0] <= SHIPYARD_LOOSE_COVER_DIST
      ]
      num_covered = len(home_yards)
      return num_covered >= 3

    # Stop strike or update strike_ship_num when ship grow.
    if (self.num_ships < 16 or self.num_ships > self.strike_ship_num or
        self.strike_ship_num <= self.num_ships - 4 or
        self.strike_success_step == self.step - 1):
      self.strike_ship_num = self.num_ships

    def should_attack_enemy_shipyard(enemy_yard):
      # Don't use bomb if ship group is small.
      if self.strike_ship_num < 18:
        return False

      # Enemy shipyard too near or within_home_boundary
      enemy_yard_dist, yard = self.get_nearest_home_yard(enemy_yard.cell)
      if (is_enemy_yard_within_home_boundary(enemy_yard, enemy_yard_dist) and
          is_enemy_weak(enemy_yard, factor=3)):
        return True

      # TODO(wangfei): use dynamic factor.
      # 18-4, 24-5, 32-6
      bomb_dist = (self.strike_ship_num -
                   16) // 8 + MIN_BOMB_ENEMY_SHIPYARD_DIST
      return (bomb_dist >= enemy_yard_dist
              and is_enemy_weak(enemy_yard, factor=2)
              and self.step - self.strike_success_step > STRIKE_COOLDOWN_STEPS)

    def shipyard_halite(yard, max_dist=4):
      cells = self.gradient_map.get_nearby_cells(yard.cell, max_dist=max_dist)
      halite = 0
      for cell in cells:
        halite += cell.halite
      return halite

    def super_strike_attack_dist():
      # return SUPER_STRIKE_ATTACK_MIN_DIST
      base = SUPER_STRIKE_ATTACK_MIN_DIST
      boost = max(0, self.num_ships - SUPER_MIN_STRIKE_SHIP_NUM) // 6
      return base + boost

    def no_work_ship_num():
      available_home_halite_cell = (self.home_halite_cell_num
                                    - self.below_keep_halite_cell_num)
      return self.strike_ship_num - available_home_halite_cell

    # logger.info(f"step = {self.step}, num_ships = {self.num_ships} no_work_ship_num = {no_work_ship_num()}")

    def should_trigger_super_strike():
      # Note: per-condition is not in normal strike.
      cooldown = self.step - self.strike_success_step

      has_enough_ship = self.num_ships > SUPER_MIN_STRIKE_SHIP_NUM
      # has_enough_ship = False
      # if self.step < 150:
        # # lower min strike ship num for the first super strike
        # has_enough_ship = (self.strike_ship_num >= 17)
      # if self.step < 230:
        # has_enough_ship = (self.strike_ship_num >= 22)
      # else:
        # has_enough_ship = (self.strike_ship_num >= 25)

      return (self.step >= 85 and
              cooldown > SUPER_STRIKE_COOLDOWN and
              has_enough_ship and
              self.num_shipyards >= 2 and
              no_work_ship_num() >= SUPER_STRIKE_MIN_NO_WORK_SHIP_NUM)

    def select_super_strike_shipyards(nearest_enemy_yard, margin=1):
      strike_halite_gain = SUPER_STRIKE_HALITE_GAIN
      if self.num_shipyards:
        strike_halite_gain = max(shipyard_halite(y) for y in self.shipyards)

      min_home_yard_dist = nearest_enemy_yard.home_yard_dist
      for enemy_yard in self.enemy_shipyards:
        if (enemy_yard.home_yard_dist <= min_home_yard_dist + margin
            and enemy_yard.halite >= strike_halite_gain):
          enemy_yard.is_super_strike = True
          yield enemy_yard

    def select_enemy_shipyard_target():
      max_halite = -9999
      nearest_enemy_yard = None
      for enemy_yard in self.enemy_shipyards:
        enemy_yard.halite = shipyard_halite(enemy_yard)
        dist, home_yard = self.get_nearest_home_yard(enemy_yard.cell)
        enemy_yard.home_yard_dist = dist
        if dist <= super_strike_attack_dist() and enemy_yard.halite > max_halite:
          max_halite = enemy_yard.halite
          nearest_enemy_yard = enemy_yard

      has_candidate = False
      for enemy_yard in self.enemy_shipyards:
        if enemy_yard.cell.is_targetd:
          continue

        # Nearby trigger condition.
        if should_attack_enemy_shipyard(enemy_yard):
          has_candidate = True
          enemy_yard.is_super_strike = False
          yield enemy_yard

      if has_candidate:
        return

      if max_halite < SUPER_STRIKE_HALITE_GAIN:
        return

      # TODO(wangfei): move out of this function
      # Super strike attack enemy shipyard
      if nearest_enemy_yard and should_trigger_super_strike():
        for enemy_yard in select_super_strike_shipyards(nearest_enemy_yard):
          yield enemy_yard

    def can_win_bomb_war(enemy_halite, enemy_ships, bomb_ships):
      enemy_spawns = [0] * (enemy_halite // self.c.spawn_cost)
      enemy_ships = sorted(enemy_spawns + [x.halite for x in enemy_ships])
      bomb_ships = sorted([x.halite for x in bomb_ships])
      i, j = 0, 0
      while i < len(enemy_ships) and j < len(bomb_ships):
        if enemy_ships[i] < bomb_ships[j]:
          j += 1
        elif enemy_ships[i] > bomb_ships[j]:
          i += 1
        else:
          # Ship crash.
          i, j = i + 1, j + 1

      # Add 1 to account for bombing the shipyard.
      return j, j + 1 < len(bomb_ships)

    def select_bomb_ships(enemy_yard):
      MAX_BOMB_SHIP_HALITE = 30
      if not enemy_yard.is_super_strike:
        MAX_BOMB_SHIP_NUM = 8
        MIN_EMPTY_SHIP_NUM = 4
      else:
        MAX_BOMB_SHIP_NUM = 8
        MIN_EMPTY_SHIP_NUM = 5

      ships = list(self.my_idle_ships)
      ships.sort(key=lambda s: self.manhattan_dist(enemy_yard, s))
      bomb_ships = []
      for ship in ships:
        # First X ships must be empty to strike the attack.
        if len(bomb_ships) < MIN_EMPTY_SHIP_NUM and ship.halite > 0:
          continue

        # Following ships. Don't send ship with too much halite.
        if ship.halite >= MAX_BOMB_SHIP_HALITE:
          continue

        bomb_ships.append(ship)

      bomb_ships = bomb_ships[:MAX_BOMB_SHIP_NUM]
      num_empty = sum(1 for b in bomb_ships if b.halite == 0)
      enemy_player_id = enemy_yard.player_id
      enemy_ships = [
          c.ship
          for c in self.gradient_map.get_nearby_cells(enemy_yard.cell,
                                                      max_dist=2)
          if c.ship_id and c.ship.player_id == enemy_player_id
      ]
      crash_num, can_win = can_win_bomb_war(enemy_yard.player.halite,
                                            enemy_ships, bomb_ships)
      # logger.info(f"  enemy_yard({enemy_yard.position}), crash_num={crash_num}, can_win={can_win}, enemy_num={len(enemy_ships)}, bomb_ships={len(bomb_ships)}, num_empty={num_empty}")
      if not can_win:
        return crash_num, [], enemy_yard
      return crash_num, bomb_ships, enemy_yard

    # Do not send bomb at beginning stage.
    if self.is_beginning_phrase:
      return

    # Stop bomb at closing phrase
    if self.is_closing_phrase:
      return

    # logger.info("bomb_enemy_shipyard at step = %s" % self.step)
    enemy_shipyards = (
        select_bomb_ships(y) for y in select_enemy_shipyard_target())
    enemy_shipyards = [x for x in enemy_shipyards if x[1]
                      ]  # If bomb ships exists

    enemy_shipyards.sort(key=lambda x: (x[2].halite, -x[2].home_yard_dist),
                         reverse=True)
    # enemy_shipyards.sort(key=lambda x: x[0])  # select shipyard with less crash first.

    # logger.info("--enemy_shipyards=%s" % len(enemy_shipyards))
    for _, bomb_ships, enemy_yard in enemy_shipyards:
      if enemy_yard.is_super_strike:
        logger.info(f"S={self.step} bomb at shipyard = {enemy_yard.id} {enemy_yard.position}, is_super_strike={enemy_yard.is_super_strike}"
                    f" home_yard_dist={enemy_yard.home_yard_dist} halite={int(enemy_yard.halite)}")
      for bomb_ship in bomb_ships:
        self.assign_task(bomb_ship, enemy_yard.cell, ShipTask.ATTACK_SHIPYARD)
        bomb_ship.is_strike_attack = True

      self.strike_shipyard = enemy_yard
      self.strike_shipyard.step = self.step
      # logger.info(f"  strike shipyard {enemy_yard.id} at {enemy_yard.position}")

      # One bomb at a time
      break

  def convert_shipyard(self):
    """Builds shipyard to maximize the total number of halite covered within
    |home_grown_cell_dist|."""
    MAX_SHIPYARD_NUM = 20

    MANHATTAN_DIST_RANGE1 = range(7, 8 + 1)
    AXIS_DIST_RANGE1 = range(3, 5 + 1)
    MANHATTAN_DIST_RANGE2 = range(6, 7 + 1)
    AXIS_DIST_RANGE2 = range(1, 6 + 1)
    MAX_SHIP_TO_SHIPYARD_DIST = 8
    SKIP_CENTER_AXIS_DIST = 4

    HALITE_CELL_PER_SHIP = 3.1
    if self.is_beginning_phrase:
      HALITE_CELL_PER_SHIP = 2.8
    # TODO(wangfei): use higher value
    elif self.step >= 120 and self.num_ships >= 22:
      HALITE_CELL_PER_SHIP = 3.3

    MIN_CONVERT_SHIP_NUM = 10

    self.halite_ratio = -1
    # No ship left.
    if not self.num_ships:
      return

    def shipyard_num_by_ship_num():
      if self.num_shipyards == 1 and self.num_ships >= MIN_CONVERT_SHIP_NUM:
        return 2

      if self.num_shipyards == 2 and self.num_ships >= 17:
        return 3

      if self.num_shipyards == 3 and self.num_ships >= 23:
        return 4

      if self.num_shipyards == 4 and self.num_ships >= 28:
        return 5

      if self.num_ships > 27:
        return min(5 + max((self.num_ships - 27) // 5, 0), MAX_SHIPYARD_NUM)
      return self.num_shipyards

    def shipyard_num_by_halite_ratio():
      # TODO: use existing value
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
        print('more shipyard: halite cell / ship =', halite_ratio)
      return num_yards

    def max_shipyard_num():
      plus = int(self.call_for_shipyard)
      return max(shipyard_num_by_ship_num(),
                 shipyard_num_by_halite_ratio()) + plus

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
      # return self.me_halite >= convert_threshold()

    def has_enemy_shipyard_nearby(cell):
      if self.num_ships >= 30:
        return False
      min_dist, min_yard = self.get_nearest_enemy_yard(cell)
      if min_yard and min_dist <= 3:
        return True
      return False

    def within_predefined_range(cell):
      if not self.me.shipyard_ids:
        return True

      dist_range = (MANHATTAN_DIST_RANGE1
                    if self.num_shipyards == 1 else MANHATTAN_DIST_RANGE2)

      self.get_nearest_home_yard(cell)  # populate cache
      for dist, yard in cell.nearest_home_yards[:2]:
        if dist not in dist_range:
          return False

        dist_x, dist_y = axis_manhattan_dists(cell.position, yard.position,
                                              self.c.size)
        axis_dist_range = (AXIS_DIST_RANGE1
                           if self.num_shipyards == 1 else AXIS_DIST_RANGE2)
        # That satisfy some axis distance constraints to make me feel safe.
        if dist_x not in axis_dist_range or dist_y not in axis_dist_range:
          return False
      return True

    def compute_convert_score_for_second(first_yard_cell, candidate_cell):
      score = 0
      cells = self.gradient_map.get_nearby_cells(candidate_cell, max_dist=4)
      for cell in cells:
        if cell.position != candidate_cell.position:
          score += cell.halite

        # Encourage covert
        dist2 = self.manhattan_dist(cell, first_yard_cell)
        if dist2 <= SHIPYARD_LOOSE_COVER_DIST:
          score += cell.halite
      return score

    def compute_convert_score(candidate_cell):
      # Special handle for the second shipyard.
      if self.num_shipyards == 1:
        s = compute_convert_score_for_second(self.shipyards[0].cell,
                                             candidate_cell)
        return s

      # Maximize the total value of halite when converting ship.
      # score = 0
      # cells = self.gradient_map.get_nearby_cells(candidate_cell, max_dist=4)
      # for cell in cells:
        # if cell.position != candidate_cell.position:
          # score += cell.halite
      # return score

      MAX_COVER_HALITE = 3

      # Maximize the halite conver by shipyard.
      score = 0

      cells = self.gradient_map.get_nearby_cells(candidate_cell,
                                                 max_dist=SHIPYARD_LOOSE_COVER_DIST)
      for cell in cells:
        if cell.position == candidate_cell.position:
          continue

        if cell.halite <= 0:
          continue

        score += 0.1
        covered = len(cell.covering_shipyards) + 1
        if covered:
          score += 0.1
        if covered >= 3:
          score += 0.2
      return score

    def is_center_zone(cell):
      center = Point(10, 10)
      dist_x, dist_y = axis_manhattan_dists(cell.position, center, self.c.size)
      return dist_x < SKIP_CENTER_AXIS_DIST and dist_y < SKIP_CENTER_AXIS_DIST

    def nominate_shipyard_positions():
      for cell in self.board.cells.values():
        # Exclude existing shipyard position (including enemy ones).
        if cell.shipyard_id:
          continue

        # Exclude died shipyard positions
        if cell.position in self.history_shipyard_positions:
          continue

        # Not convert too near enemy shipyard.
        if has_enemy_shipyard_nearby(cell):
          # print(f"c[{cell.position.x}, {cell.position.y}]: too near enemy")
          continue

        # Skip center zone
        if self.num_shipyards <= 4 and is_center_zone(cell):
          continue

        if (self.strike_success_position and
            self.step - self.strike_success_step <= STRIKE_CALL_FOR_SHIPYARD_STEPS
            and self.manhattan_dist(cell, self.board[self.strike_success_position]) <= 2):
          min_dist, _ = self.get_nearest_home_yard(cell)
          if min_dist <= 5:
            continue
          cell.convert_score = compute_convert_score(cell)
          # logger.info(f"strike position candidate: c[{cell.position.x}, {cell.position.y}]")
          yield cell
          continue

        if not within_predefined_range(cell):
          # print(f"c[{cell.position.x}, {cell.position.y}]: not in defined range")
          continue

        # Have a nearby ship.
        dist_to_yard, _ = self.find_nearest_enemy(cell, self.ships)
        if dist_to_yard > MAX_SHIP_TO_SHIPYARD_DIST:
          # print(f"c[{cell.position.x}, {cell.position.y}]: no near by ships")
          continue

        cell.convert_score = compute_convert_score(cell)
        yield cell

    def convert_ship(ship):
      self.cost_halite += (self.c.convert_cost - ship.halite)
      ship.next_action = ShipAction.CONVERT
      ship.has_assignment = True
      ship.cell.is_targetd = True

    def call_for_ship(cell, halite_check=True):
      ships = sorted(self.my_idle_ships,
                     key=lambda s: self.manhattan_dist(s, cell))
      for ship in ships:
        if halite_check and not has_enough_halite(ship):
          continue

        dist_to_yard = self.manhattan_dist(ship, cell)
        # Annoy nearby enemy.
        min_enemy_to_yard_dist, min_enemy = self.find_nearest_enemy(
            cell, self.enemy_ships)
        if (min_enemy and min_enemy_to_yard_dist <= dist_to_yard and
            min_enemy.halite < ship.halite):
          continue

        if (halite_check == True and ship.position == cell.position and
            not self.has_enemy_nearby(ship.cell)):
          convert_ship(ship)
          self.call_for_shipyard = False
          return True

        if ship.position != cell.position:
          print("Send ship(%s %s) to shipyard position (%s), dist=%s" %
                (ship.id, ship.position, cell.position, dist_to_yard))
          # Let's use GOTO_HALITE for now.
          self.assign_task(ship, cell, ShipTask.INITIAL_SHIPYARD)
          return True

      return False

    # Pre cash money for shipyard conversion when moving towards.
    self.save_for_converting = self.c.convert_cost

    candidate_cells = list(nominate_shipyard_positions())
    # logger.info("Convert shipyard: nominate cells count = %s" %len(candidate_cells))
    if not candidate_cells:
      return

    candidate_cells.sort(key=lambda c: c.convert_score, reverse=True)
    for cell in candidate_cells:
      if call_for_ship(cell):
        # Send one more ship for backup and protect.
        call_for_ship(cell, halite_check=False)

        # One shipyard at a time.
        return

  def compute_ship_moves(self):
    """Computes ship moves to its target.

    Maximize total expected value.
    * prefer the move to the target (distance is shorter)
    * not prefer move into enemy with lower halite (or equal)
    """
    spawn_cost = self.board.configuration.spawn_cost
    convert_cost = self.board.configuration.convert_cost
    collect_rate = self.board.configuration.collect_rate

    def compute_weight(ship, next_position):
      ignore_neighbour_cell_enemy = False
      target_cell = ship.target_cell
      next_cell = self.board[next_position]

      # If a non-followed ship's next move is to alley SPAWNING shipyard, skip
      yard = next_cell.shipyard
      if (yard and yard.player_id == self.me.id and
          yard.next_action == ShipyardAction.SPAWN and
          not ship.is_followed):
        return MIN_WEIGHT

      wt = 0
      dist = manhattan_dist(next_position, target_cell.position, self.c.size)

      # Retreat ship in danger.
      if (ship.is_in_danger and
          ship.task_type not in (ShipTask.ATTACK_SHIP, ShipTask.ATTACK_SHIPYARD)):
        # TODO: how about ship.halite > 0 but is attacking. should it retreat?
        wt -= ship.enemy_gradient[next_position.x, next_position.y]
      else:
        # add weight for the cell that's nearer to the destination.
        ship_dist = self.manhattan_dist(ship, target_cell)
        wt = ship_dist - dist

      # When attacking, do not stay on halite cell
      if (ship.task_type in (ShipTask.ATTACK_SHIP, ShipTask.ATTACK_SHIPYARD) and
          ship.position == next_position and ship.cell.halite > 0):
        wt -= 2000

      # Do not stay when followed
      if (ship.task_type in (ShipTask.RETURN,) and
          hasattr(ship, "followers") and ship.position == next_position):
        wt -= 2000

      # Try not move onto halite cells.
      if next_cell.halite > 0 and next_position != target_cell.position:
        halite_gradient = self.halite_gradient[next_position.x, next_position.y]
        wt -= halite_gradient * 0.1

      # If collecting halite
      if ((ship.task_type == ShipTask.GOTO_HALITE or
           ship.task_type == ShipTask.COLLECT) and target_cell.halite > 0):
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
        if hasattr(ship, 'followers'):
          wt += self.c.spawn_cost

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
        # Only ignore enemy when the ship is on the yard.
        if next_position == target_cell.position:
          ignore_neighbour_cell_enemy = True

      def move_away_from_enemy(enemy, ship, side_by_side=True):
        """Collides with enemy if my ship has less halite."""
        if ship.halite < enemy.halite:
          return False
        elif ship.halite > enemy.halite:
          return True

        # enemy.halite == ship.halite
        assert enemy.halite == ship.halite
        if ship.halite > 0:
          return True

        if getattr(enemy, 'within_home_boundary', False):
          avoid_rate = 0.8 if self.num_ships >= 28 else AVOID_COLLIDE_RATIO
        else:
          if getattr(ship, 'is_strike_attack', False):
            avoid_rate = 0.95
          else:
            avoid_rate = 1.0

        return random.random() < avoid_rate

      # If there is an enemy in next_position with lower halite
      if has_enemy_ship(next_cell, self.me):
        # If there is an enemy sitting on its shipyard, collide with him.
        if (ship.task_type == ShipTask.ATTACK_SHIPYARD and ship.halite == 0 and
            dist == 0):
          pass
        elif move_away_from_enemy(next_cell.ship, ship, side_by_side=True):
          wt -= (spawn_cost + ship.halite)

      # If there is an enemy in neighbor next_position with lower halite
      if not ignore_neighbour_cell_enemy:
        for nb_cell in get_neighbor_cells(next_cell):
          # If there is an enemy sitting on its shipyard, ignore it
          if (ship.task_type == ShipTask.ATTACK_SHIPYARD and
              ship.halite == 0 and dist == 1 and
              nb_cell.position == target_cell.position):
            continue

          if has_enemy_ship(nb_cell, self.me):
            if move_away_from_enemy(nb_cell.ship, ship, side_by_side=False):
              wt -= (spawn_cost + ship.halite)
      return wt

    # Skip only convert ships.
    ships = [s for s in self.ships if not s.next_action]
    next_positions = {
        make_move(s.position, move, self.c.size)
        for s in ships
        for move in POSSIBLE_MOVES
    }

    # Duplicate position at final stage for more halite.
    if self.step + 7 >= self.c.episode_steps:

      def duplicate_positions(positions):
        for p in positions:
          cell = self.board[p]
          if cell.shipyard_id and cell.shipyard.player_id == self.me.id:
            # Each positions can only accept 4 incoming moves at maximum.
            for _ in range(4):
              yield p
          else:
            yield p

      next_positions = list(duplicate_positions(next_positions))

    def get_position_to_index():
      d = defaultdict(list)
      for i, pos in enumerate(next_positions):
        d[pos].append(i)
      return d

    position_to_index = get_position_to_index()
    C = np.ones((len(ships), len(next_positions))) * MIN_WEIGHT
    for ship_idx, ship in enumerate(ships):
      for move in POSSIBLE_MOVES:
        next_position = make_move(ship.position, move, self.c.size)
        for poi_idx in position_to_index[next_position]:
          poi_idx = position_to_index[next_position]
          C[ship_idx, poi_idx] = compute_weight(ship, next_position)

    rows, cols = scipy.optimize.linear_sum_assignment(C, maximize=True)

    index_to_position = list(next_positions)
    for ship_idx, poi_idx in zip(rows, cols):
      ship = ships[ship_idx]
      next_position = index_to_position[poi_idx]

      ship.next_cell = self.board[next_position]
      ship.next_action = direction_to_ship_action(ship.position, next_position,
                                                  self.c.size)
      # print(ship.id, 'at', ship.position, 'goto', next_position)

    if len(rows) != len(ships):
      matched_ship_ids = set()
      for ship_idx in rows:
        matched_ship_ids.add(ships[ship_idx].id)

      for ship in ships:
        print('ship %s (matchd=%s), at %s, has_assignment=%s, task=%s' %
              (ship.id, ship.id in matched_ship_ids, ship.position,
               ship.has_assignment, ship.task_type))
        for move in POSSIBLE_MOVES:
          next_position = make_move(ship.position, move, self.c.size)
          wt = compute_weight(ship, next_position)
          print('   to %s, wt=%.2f' % (next_position, wt))
    # assert len(rows) == len(ships), "match=%s, ships=%s" % (len(rows),
    # len(ships))

  def spawn_ships(self):
    """Spawns farmer ships if we have enough money and no collision with my own
    ships."""
    SHIP_NUM_HARD_LIMIT = 100

    # When leading, convert as much as possible.
    def max_ship_num():
      return SHIP_NUM_HARD_LIMIT

    def spawn(yard):
      self.cost_halite += self.c.spawn_cost
      yard.next_action = ShipyardAction.SPAWN

    def spawn_threshold():
      return self.save_for_converting + self.c.spawn_cost

    # TODO(wangfei): Test collect ratio
    def ship_collect_steps(t, margin=5, collect_step_ratio=0.3):
      return int(max(t - margin, 0) * collect_step_ratio)

    def expect_spawn_ship_num():
      total_halite = int(self.total_collectable_halite)
      total_required_time = self.total_collect_steps

      remain_steps = self.c.episode_steps - self.step
      ship_steps = ship_collect_steps(remain_steps)
      total_ship_time = self.num_ships * ship_steps

      avg_step_gain = int(total_halite / (total_required_time + 0.1))

      expect_ship_num = 0
      ship_contibue_steps = 0
      target_steps = total_required_time - total_ship_time
      for i in range(1, self.num_shipyards+1):
        left_steps = target_steps - ship_contibue_steps
        ship_gain = min(ship_steps, left_steps) * avg_step_gain - self.c.spawn_cost
        if ship_gain < 0:
          break

        expect_ship_num += 1
        ship_contibue_steps += ship_steps

      # logger.info(f"step={self.step} total_halite={total_halite}, total_required_time={total_required_time}"
                  # f" total_ship_time={total_ship_time}"
                  # f" avg_step_gain={avg_step_gain}, target_steps={target_steps},"
                  # f" expect_ship_num={expect_ship_num}")

      # Enough number of ships
      if total_ship_time >= total_required_time:
        return 0

      return expect_ship_num

    # Too many ships. (hard limit)
    if self.num_ships >= max_ship_num():
      return

    # No more ships after X
    if self.num_ships >= 3 and self.step >= 320:
      return

    max_spawn_num = None
    if self.step >= 300:
      max_spawn_num = expect_spawn_ship_num()

    random.shuffle(self.shipyards)
    for shipyard in self.shipyards[:max_spawn_num]:
      # Stop when not enough halite left.
      if self.num_ships and self.me_halite < spawn_threshold():
        break

      spawn(shipyard)

  def final_stage_back_to_shipyard(self):
    MARGIN_STEPS = 7
    MIN_HALITE_TO_YARD = 10

    def ship_and_dist_to_yard():
      for ship in self.my_idle_ships:
        if ship.halite <= MIN_HALITE_TO_YARD:
          continue
        _, yard = self.get_nearest_home_yard(ship.cell)
        if yard:
          dist = self.manhattan_dist(ship, yard)
          yield dist, ship, yard

    if not self.me.shipyard_ids:
      return

    ship_dists = list(ship_and_dist_to_yard())
    if not ship_dists:
      return

    for min_dist, ship, min_dist_yard in ship_dists:
      if self.step + min_dist + MARGIN_STEPS > self.c.episode_steps:
        self.assign_task(ship, min_dist_yard.cell, ShipTask.RETURN)

  def spawn_if_shipyard_in_danger(self):
    """Spawn ship if enemy nearby my shipyard and no ship's next_cell on this
    shipyard."""
    if self.is_closing_phrase:
      return

    ship_next_positions = {
        ship.next_cell.position
        for ship in self.ships
        if ship.next_action != ShipAction.CONVERT
    }

    def is_shipyard_in_danger(yard):
      # If there is one of my ship will be the on yard in the next round.
      if yard.position in ship_next_positions:
        return False
      return yard.is_in_danger

    def spawn(yard):
      self.cost_halite += self.c.spawn_cost
      yard.next_action = ShipyardAction.SPAWN

    for yard in self.shipyards:
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
      print('  p[%s](s=%s, y=%s, h=%s, c=%s, mc=%s)' %
            (player.id, num_ships, num_shipyards, player.halite, cargo(player),
             mean_cargo(player)),
            end=end)

    def print_ship_task_type():
      task = Counter()
      for ship in self.ships:
        task[ship.task_type] += 1
      items = sorted(task.items(), key=lambda x: x[0].name)
      print(", ".join("%s=%s(%.0f%%)" % (k.name, v, v / self.num_ships * 100)
                      for k, v in items))

    print(
        '#%s' % self.step, 'halite(n=%s, mean=%s)' %
        (len(self.halite_cells), int(
            self.mean_halite_value)),
        'home_halite=(d=%s, cover=%.0f%%, n=%s, m=%s, n/s=%.1f)' %
        (self.home_grown_cell_dist, self.num_home_halite_cells /
         len(self.halite_cells) * 100, self.num_home_halite_cells,
         int(self.mean_home_halite), self.halite_ratio))
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

    to mimic the we don't want back home.
    """
    enemy_carry = 0
    halite_left = poi.halite
    if ship and has_enemy_ship(poi, self.me):
      # Dist to move to neighour cell of POI.
      dist = max(0, ship_to_poi - 1)

      # Halite will decrease if there is ship sitting on it.
      halite_left = poi.halite * HALITE_RETENSION_BY_DIST[dist]

      # Give up if my ship has more halite then enemy.
      enemy = poi.ship
      enemy_halite = poi.ship.halite + int(poi.halite - halite_left)
      if ship and ship.halite >= enemy_halite:
        return MIN_WEIGHT

      enemy_carry = enemy.halite

    carry = ship and ship.halite or 0
    travel = ship_to_poi + poi_to_yard
    opt_steps = optimal_mining_steps(carry, poi.halite, travel)
    if opt_steps < min_mine:
      opt_steps = min_mine

    total_halite = (carry + enemy_carry +
                    (1 - HALITE_RETENSION_BY_DIST[opt_steps]) * halite_left)
    return total_halite / (ship_to_poi + opt_steps + poi_to_yard / 3)

  def get_trapped_enemy_ships(self, max_attack_num):
    """A enemy is trapped if there're at least one ship in each quadrant."""
    if self.is_final_phrase:
      return

    # Do not attack enemy during ending.
    adjust = 0
    if self.num_ships >= 25:
      adjust = 1
    MAX_ATTACK_DIST = 4 + adjust

    MIN_ATTACK_QUADRANT_NUM = 3
    # if self.step >= 100:
    # MIN_ATTACK_QUADRANT_NUM -= 1
    if self.num_ships >= 35:
      MIN_ATTACK_QUADRANT_NUM -= 1

    def is_enemy_within_home_boundary(enemy):
      """1. Within distance of 2 of any shipyard
         2. triple covered by multiple shipyards.
      """
      covered = 0
      self.get_nearest_home_yard(enemy.cell)  # populate cache.
      for dist, yard in enemy.cell.nearest_home_yards:
        if dist <= SHIPYARD_TIGHT_COVER_DIST + 1:
          return True
        if dist <= SHIPYARD_LOOSE_COVER_DIST:
          covered += 1
      return covered >= 3

    def get_attack_ships(enemy):
      # Extra attack distance for enemy within home boundary.
      max_attack_dist = MAX_ATTACK_DIST
      if enemy.within_home_boundary:
        max_attack_dist = max(5, max_attack_dist + 1)

      for ship in self.my_idle_ships:
        dist = self.manhattan_dist(ship, enemy)
        if dist <= max_attack_dist and ship.halite < enemy.halite:
          yield dist, ship

    def annotate_by_quadrant(dist_ships, enemy):
      """Sort to make sure at least one ship is selected in each quadrant."""
      quadrants = set()
      for dist, ship in dist_ships:
        q = get_quadrant(ship.position - enemy.position)
        q_exist = int(q in quadrants)
        quadrants.add(q)
        yield (q_exist, dist), ship

    # Collect follower enemy ships.
    # enemy_follower_ids = set()
    # for ship in self.me.ships:
    # followers = getattr(ship, 'followers', None)
    # if followers:
    # for f in followers:
    # enemy_follower_ids.add(f.id)

    for enemy in self.enemy_ships:
      enemy.within_home_boundary = is_enemy_within_home_boundary(enemy)
      # enemy.is_follower = (enemy.id in enemy_follower_ids)

      dist_ships = get_attack_ships(enemy)
      dist_ships = list(annotate_by_quadrant(dist_ships, enemy))
      dist_ships.sort(key=lambda x: x[0])
      quadrant_num = len({
          get_quadrant(ship.position - enemy.position) for _, ship in dist_ships
      })

      # Reduce quadrant_num for home boundary enemy.
      min_attack_quadrant_num = MIN_ATTACK_QUADRANT_NUM
      # if enemy.within_home_boundary:
      # min_attack_quadrant_num -= 1

      if quadrant_num >= min_attack_quadrant_num:
        enemy.quadrant_num = quadrant_num
        enemy.attack_ships = [ship for _, ship in dist_ships][:max_attack_num]
        yield enemy
      # elif enemy.is_follower and len(dist_ships) > 0:
      # enemy.attack_ships = [ship for _, ship in dist_ships][:2]
      # yield enemy

  def get_ship_halite_pairs(self, ships, halites):
    for ship_idx, ship in enumerate(ships):
      for poi_idx, cell in enumerate(halites):
        # Do not go to halite with too many enemy around.
        enemy_count = self.gradient_map.count_nearby_true_enemy(cell, ship)
        if enemy_count >= MIN_ENEMY_TO_RUN:
          continue
        yield ship_idx, poi_idx

  def get_rescue_escape_ship_pairs(self, ships):

    def is_supporter(sup, followers, followed_ship):
      if sup.is_followed and sup.halite <= 10:
        return False

      min_dist1, _ = self.get_nearest_home_yard(followed_ship.cell)
      min_dist2, _ = self.get_nearest_home_yard(sup.cell)
      if min_dist2 >= min_dist1:
        return False

      for follower in followers:
        if sup.halite >= follower.halite:
          return False
      return True

    for ship in ships:
      if not ship.is_followed:
        continue

      supporters = []
      for sup in ships:
        dist = self.manhattan_dist(ship, sup)
        if dist > ALLEY_SUPPORT_DIST:
          continue

        if is_supporter(sup, ship.followers, ship):
          supporters.append(sup)

      supporters.sort(key=lambda sup: self.manhattan_dist(ship, sup))
      yield ship, supporters[:MAX_SUPPORT_NUM]

  def get_rescue_attack_ship_pairs(self, ship_supporters):
    for ship, supporters in ship_supporters:
      for sup in supporters:
        for follower in ship.followers:
          yield sup, follower

  def optimal_assignment(self):
    ATTACK_PER_ENEMY = 7
    SHIPYARD_DUPLICATE_NUM = 5

    def shipyard_duplicate_num():
      if self.is_final_phrase:
        return 1
      return SHIPYARD_DUPLICATE_NUM

    ships = list(self.my_idle_ships)

    halites = [c for c in self.halite_cells if c.halite >= c.keep_halite_value]
    ship_halite_pairs = set(self.get_ship_halite_pairs(ships, halites))

    # Shipyards is duplicated to allow multiple ships having a same target.
    shipyards = [y.cell for y in self.shipyards] * shipyard_duplicate_num()

    # Attack enemy.
    trapped_enemy_ships = list(self.get_trapped_enemy_ships(ATTACK_PER_ENEMY))
    enemy_cells = [e.cell for e in trapped_enemy_ships] * ATTACK_PER_ENEMY
    attack_pairs = {
        (s.id, e.id) for e in trapped_enemy_ships for s in e.attack_ships
    }

    # Guard shipyard.
    offended_shipyards = list(self.get_offended_shipyard())
    offended_cells = [y.cell for y, _ in offended_shipyards]
    guard_paris = {
        (s.id, y.id) for y, ships in offended_shipyards for s in ships
    }

    # Support alley ship escape
    ship_supporters = list(self.get_rescue_escape_ship_pairs(ships))
    ship_supporter_pairs = {
        (s.id, sup.id) for s, sups in ship_supporters for sup in sups
    }
    support_attack_pairs = {
        (s.id, e.id)
        for s, e in self.get_rescue_attack_ship_pairs(ship_supporters)
    }

    # Dedup supporters
    supporter_ids = {sup.id for _, sups in ship_supporters for sup in sups}
    supporters = [
        sup for _, sups in ship_supporters for sup in sups
        if sup.id in supporter_ids
    ]
    # Do not goto supporters.
    supporters = []

    pois = halites + shipyards + enemy_cells + supporters + offended_cells

    def is_halite_column(x):
      return x < len(halites)

    def is_shipyard_column(x):
      return len(halites) <= x < len(halites) + len(shipyards)

    def is_enemy_column(x):
      return (len(halites) + len(shipyards) <= x and
              x < len(halites) + len(shipyards) + len(enemy_cells))

    def is_supporter_column(x):
      left = len(halites) + len(shipyards) + len(enemy_cells)
      return (left <= x and x < left + len(supporters))

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
          if (i, j) not in ship_halite_pairs:
            v = MIN_WEIGHT
          else:
            # If the target is a halite cell, with enemy considered.
            v = self.halite_per_turn(ship, poi, ship_to_poi, poi_to_yard)
        elif is_shipyard_column(j):
          v = 0
          # If the target is a shipyard.
          if ship_to_poi > 0:
            v += ship.halite

          # If have follower, let the followed ship back.
          if ship.is_followed or ship.is_in_danger:
            v += self.c.spawn_cost

          # Force send ship home.
          if ship.halite >= MAX_SHIP_CARGO and not self.is_closing_phrase:
            v += ship.halite

          v /= (ship_to_poi or 1)
        elif is_enemy_column(j):
          # If attack enemy
          enemy = poi.ship
          v = MIN_WEIGHT  # not exists edge.
          attack_key = (ship.id, enemy.id)
          if attack_key in attack_pairs or attack_key in support_attack_pairs:
            # v = (self.c.spawn_cost + enemy.halite + enemy.cell.halite) / ship_to_poi
            bonus = enemy.cell.halite
            if getattr(enemy, 'within_home_boundary', False):
              bonus = +100
            if attack_key in support_attack_pairs:
              bonus += self.c.spawn_cost
            v = (self.c.spawn_cost + enemy.halite + bonus) / ship_to_poi
        elif is_supporter_column(j):
          # Ship being followerd goes to alley ship.
          if (ship.id, poi.id) in ship_supporter_pairs:
            v = (ship.halite + self.c.spawn_cost) / ship_to_poi
        else:
          # If shipyard is offended.
          yard = poi.shipyard
          v = MIN_WEIGHT
          if (ship.id, yard.id) in guard_paris:
            v = (self.c.spawn_cost + self.c.convert_cost +
                 ship.halite) / (ship_to_poi or 1)

            # If selected as guard ship, the followed ship has priority.
            if hasattr(ship, 'followers'):
              v += self.c.spawn_cost

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
      elif is_supporter_column(poi_idx):
        # Reuse return task
        task_type = ShipTask.RETURN
      else:
        task_type = ShipTask.GUARD_SHIPYARD
        shipyard = poi_cell.shipyard
        shipyard.is_in_danger = False
        enemy = shipyard.offend_enemy
        # print('guide task: ', ship.position, poi_cell.position,
        # shipyard.offend_enemy.position)

      self.assign_task(ship, poi_cell, task_type, enemy=enemy)

  def get_offended_shipyard(self):
    """Guard shipyard."""

    def shipyard_defend_dist():
      # if len(self.me.shipyard_ids) > 1 or self.me_halite >= self.c.spawn_cost:
      # return 3
      return 4

    def offend_enemy_ships(yard):
      not_enough_halite_to_spawn = self.me_halite < self.c.spawn_cost
      for enemy in self.enemy_ships:
        if self.manhattan_dist(enemy, yard) > shipyard_defend_dist():
          continue

        # If the enemy has money, then I'll just let it send it for me.
        if not_enough_halite_to_spawn or enemy.halite == 0:
          yield enemy

    def get_defend_ships(yard, enemy, enemy_to_yard_dist):
      for ship in self.my_idle_ships:
        dist_to_yard = self.manhattan_dist(ship, yard)
        if (dist_to_yard < enemy_to_yard_dist or
            (dist_to_yard == enemy_to_yard_dist and
             (enemy.halite > 0 and ship.halite < enemy.halite or
              enemy.halite == 0 and ship.halite == 0))):
          # print('defend enemy(%s) by ship(%s, %s)' % (enemy.position, ship.id, ship.position))
          yield ship

    for yard in self.shipyards:
      yard.is_in_danger = False
      min_enemy_dist, enemy = self.find_nearest_enemy(yard.cell,
                                                      offend_enemy_ships(yard))
      if enemy is None:
        continue
      # No need guard shipyard if enemy has halite (by turn order, spawn comes
      # before collision)
      if yard.next_action == ShipyardAction.SPAWN and enemy.halite > 0:
        continue

      yard.is_in_danger = True
      defend_ships = list(get_defend_ships(yard, enemy, min_enemy_dist))
      if defend_ships:
        yard.offend_enemy = enemy
        yield yard, defend_ships

  def update_ship_follower(self):
    """If a ship is followed by enemy, send it back home."""
    for ship in self.ships:
      ship.is_followed = False
      ship.is_in_danger = False
      ship.enemy_gradient = None

    for ship in self.my_idle_ships:
      # Update follower info.
      ship.is_followed = self.follower_detector.is_followed(ship)
      if ship.is_followed:
        ship.followers = self.follower_detector.get_followers(ship)

      # Update enemy surranding.
      enemy_count = self.gradient_map.count_nearby_true_enemy(ship.cell, ship,
                                                              with_cell_halite=False)
      if enemy_count >= MIN_ENEMY_TO_RUN:
        ship.enemy_gradient = self.gradient_map.get_enemy_gradient(ship.cell,
                                                                  halite=ship.halite,
                                                                  broadcast_dist=3,
                                                                  max_dist=3,
                                                                  normalize=True)
        ship.is_in_danger = True


      # Not assign task explicitly, but using optimal assignment for it.
      # self.assign_task(ship, yard.cell, ShipTask.RETURN)
      # print('ship(%s) at %s is followed by enemy(%s) at %s by %s times' %
      # (ship.id, ship.position, ship.follower.id, ship.follower.position,
      # self.follower_detector.follow_count[ship.id]))

  def clear_spawn_ship(self):
    """Clear ship spawn for ship to return homeyard with follower."""

    def is_my_shipyard_spawning(cell):
      return (cell.shipyard_id and cell.shipyard.player_id == self.me.id and
              cell.shipyard.next_action == ShipyardAction.SPAWN)

    for ship in self.ships:
      cell = ship.next_cell
      if cell and is_my_shipyard_spawning(cell):
        cell.shipyard.next_action = None

  def convert_trapped_ship_to_shipyard(self):

    def is_ship_trapped(ship):
      enemy_nearby_count = 0
      danger_cell_count = 0
      enemy_gradient = self.gradient_map.get_enemy_gradient(ship.cell,
                                                            halite=ship.halite,
                                                            broadcast_dist=1,
                                                            max_dist=2)
      for cell in get_neighbor_cells(ship.cell):
        if has_enemy_ship(cell, self.me):
          enemy = cell.ship
          if enemy.halite < ship.halite:
            enemy_nearby_count += 1
            continue

        if enemy_gradient[cell.position.x,
                          cell.position.y] >= self.c.spawn_cost:
          danger_cell_count += 1

      return (enemy_nearby_count == 4 or
              (enemy_nearby_count == 3 and danger_cell_count == 1))

    def has_enough_halite(ship):
      return ship.halite + self.me_halite >= self.c.convert_cost

    for ship in self.ships:
      if (ship.halite >= 100 and ship.next_action != ShipAction.CONVERT and
          is_ship_trapped(ship) and has_enough_halite(ship)):
        ship.next_action = ShipAction.CONVERT
        self.cost_halite += (self.c.convert_cost - ship.halite)
        print("Convert ship in danger %s at %s h=%s for trapped." %
              (ship.id, ship.position, ship.halite))

  def execute(self):
    self.save_for_converting = 0
    self.collect_game_info()

    if self.first_shipyard_set:
      self.convert_shipyard()
      self.update_ship_follower()
      self.spawn_ships()

      self.bomb_enemy_shipyard()

      self.final_stage_back_to_shipyard()
      self.optimal_assignment()
    else:
      self.convert_first_shipyard()

    self.compute_ship_moves()
    self.spawn_if_shipyard_in_danger()

    self.clear_spawn_ship()
    self.convert_trapped_ship_to_shipyard()
    if not self.simulation:
      self.print_info()


STRATEGY = ShipStrategy()


@board_agent
def agent(board):
  STRATEGY.update(board)
  STRATEGY.execute()
