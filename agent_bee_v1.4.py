#!/usr/bin/env python
"""
1. Fix send bomb when I have only a few ships.

Tournament - ID: Ki5jnE, Name: Your Halite 4 Trueskill Ladder | Dimension - ID: bq5kgn, Name: Halite 4 Dimension
Status: running | Competitors: 8 | Rank System: trueskill

Total Matches: 684 | Matches Queued: 62
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches
bee v1.4                       | AKSI4CRQduly   | 38.8734341      | μ=41.590, σ=0.906  | 288
bee v1.3                       | hmzmPDJBbjrh   | 36.7828514      | μ=39.280, σ=0.832  | 274
swarm                          | ivIufMVdUC96   | 29.5216172      | μ=31.671, σ=0.716  | 297
v3.3 no min                    | JNx5yecvaHFH   | 28.6792750      | μ=30.841, σ=0.721  | 325
v3.1                           | x2JUzNPqFDy6   | 27.7322780      | μ=29.897, σ=0.722  | 362
v2.2.1                         | UN7SNKKVzyYW   | 24.4781455      | μ=26.623, σ=0.715  | 390
manhattan                      | s8U447capUik   | 17.1213816      | μ=19.472, σ=0.784  | 415
v1.2                           | MiSa2eusdnUa   | 15.2531972      | μ=17.676, σ=0.808  | 385
"""

import random
import timeit
import logging
from enum import Enum, auto

import networkx as nx
import numpy as np
from kaggle_environments.envs.halite.helpers import *

MIN_WEIGHT = -99999

BEGINNING_PHRASE_END_STEP = 40
ENDING_PHRASE_STEP = 320

# If my halite is less than this, do not build ship or shipyard anymore.
MIN_HALITE_TO_BUILD_SHIPYARD = 1000
MIN_HALITE_TO_BUILD_SHIP = 1000

# The factor is num_of_ships : num_of_shipyards
SHIP_TO_SHIYARD_FACTOR = 100

# To control the mining behaviour
MIN_HALITE_BEFORE_HOME = 100
MIN_HALITE_FACTOR = 3

# Controls the number of ships.
MAX_SHIP_NUM = 23
MAX_DEFEND_SHIPS = 14

# Threshold for attack enemy nearby my shipyard
TIGHT_ENEMY_SHIP_DEFEND_DIST = 5
LOOSE_ENEMY_SHIP_DEFEND_DIST = 7

# Threshod used to send bomb to enemy shipyard
MIN_ENEMY_YARD_TO_MY_YARD = 7

# Threshold used to estimate best cell for shipyard.
NEARBY_HALITE_CELLS_DIST = 5

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

  def __init__(self, board):
    self.board = board
    self.me = board.current_player

    # Init halite cells
    self.halite_cells = []
    for cell in board.cells.values():
      cell.is_targetd = False
      if cell.halite > 0:
        self.halite_cells.append(cell)

    # Default ship to stay on the same cell without assignment.
    for ship in self.me.ships:
      ship.has_assignment = False
      ship.target_cell = ship.cell
      ship.next_cell = ship.cell
      ship.task_type = ShipTask.STAY_ON_CELL_TASK

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
  def num_ships(self):
    return len(self.me.ship_ids)

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
  def assign_task(ship, target_cell: Cell, task_type: ShipTask):
    """Add a task to a ship."""
    ship.has_assignment = True
    ship.target_cell = target_cell
    ship.task_type = task_type
    ship.target_cell.is_targetd = True

  def collect_game_info(self):
    self.mean_halite_value = MIN_HALITE_BEFORE_HOME
    if self.halite_cells:
      halite_values = [c.halite for c in self.halite_cells]
      self.mean_halite_value = np.mean(halite_values)

    self.max_enemy_halite = -1
    self.max_enemy_id = None
    for p in self.board.opponents:
      if p.halite >= self.max_enemy_halite:
        self.max_enemy_halite = p.halite
        self.max_enemy_id = p.id

  def max_expected_return_cell(self, ship):
    HOME_GROWN_CELL_DIST = 6
    MIN_STOP_COLLECTIONG_THRESHOLD = 10.0

    # If less than this value, Give up mining more halite from this cell.
    CELL_STOP_COLLECTING_HALITE = 200.0
    CELL_START_COLLECTING_HALITE = 300.0

    def is_home_grown_halite_cell(cell):
      _, min_yard = self.find_nearest_shipyard(ship, self.me.shipyards)
      if min_yard:
        cell_to_yard_dist = manhattan_dist(cell.position, min_yard.position,
                                           self.c.size)
        if cell_to_yard_dist <= HOME_GROWN_CELL_DIST:
          return True
      return False

    def stop_threshold(cell):
      threshold = MIN_STOP_COLLECTIONG_THRESHOLD
      if is_home_grown_halite_cell(cell):
        threshold = CELL_STOP_COLLECTING_HALITE / 2
        if (BEGINNING_PHRASE_END_STEP < self.step < ENDING_PHRASE_STEP and
            self.num_ships >= 10):
          threshold = CELL_STOP_COLLECTING_HALITE
      return threshold

    max_cell = None
    max_expected_return = 0
    for c in self.halite_cells:
      if c.is_targetd or c.halite < stop_threshold(c):
        continue

      expect_return = self.compute_expect_halite_return(ship, ship.position, c)
      if expect_return > max_expected_return:
        max_cell = c
        max_expected_return = expect_return
    return max_expected_return, max_cell

  def find_nearest_shipyard(self, ship, shipyards):
    min_dist = 99999
    min_dist_yard = None
    for y in shipyards:
      d = manhattan_dist(ship.position, y.position, self.c.size)
      if d < min_dist:
        min_dist = d
        min_dist_yard = y
    return min_dist, min_dist_yard

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
    threshold = int(
        max(self.mean_halite_value * MIN_HALITE_FACTOR, MIN_HALITE_BEFORE_HOME))
    for ship in self.my_idle_ships:
      if ship.halite < threshold:
        continue

      _, min_dist_yard = self.find_nearest_shipyard(ship, self.me.shipyards)
      if min_dist_yard:
        self.assign_task(ship, min_dist_yard.cell,
                         ShipTask.RETURN_TO_SHIPYARD_TASK)

  def compute_expect_halite_return(self, ship, position, target_cell):
    """Position maybe other than ship.position in case of computing moves"""
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
      expected_halite = min(target_cell.halite * (self.growth_factor**dist),
                            self.board.configuration.max_cell_halite)
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

      home_dist, min_dist_yard = self.find_nearest_shipyard(
          ship, self.me.shipyards)
      home_return = ship.halite / (home_dist + 1)

      if home_return > halite_return:
        self.assign_task(ship, min_dist_yard.cell,
                         ShipTask.RETURN_TO_SHIPYARD_TASK)
      else:
        self.assign_task(ship, max_cell, ShipTask.GOTO_HALITE_TASK)

  def attack_enemy_yard(self):
    """Having enough farmers, let's send ghost to enemy shipyard."""

    def max_bomb_dist():
      # Don't use bomb if ship group is small.
      if self.num_ships < 10:
        return 0

      # If having enough money and halite.
      if (len(self.me.ship_ids) >= MAX_SHIP_NUM and
          self.me.halite > self.max_enemy_halite + self.c.spawn_cost) * 2:
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
      min_dist_ship = None
      for ship in self.my_idle_ships:
        # Don't send halite to enemy.
        if ship.halite > 10:
          continue

        dist = manhattan_dist(enemy_yard.position, ship.position, self.c.size)
        if dist < min_dist:
          min_dist = dist
          min_dist_ship = ship
      return min_dist, min_dist_ship

    # TODO: sort enemy shipyard?
    for y in non_targeted_enemy_shipyards():
      _, min_dist_ship = select_bomb_ship(y)
      if min_dist_ship:
        self.assign_task(min_dist_ship, y.cell,
                         ShipTask.DESTORY_ENEMY_YARD_TASK)

        # One bomb at a time
        break

  def attack_enemy_ship(self):
    """Send ship to enemy to protect my shipyard."""

    def all_enemy_ships(defend_distance):
      for opp in self.board.opponents:
        for s in opp.ships:
          min_dist, min_yard = self.find_nearest_shipyard(s, self.me.shipyards)
          if min_yard and min_dist <= defend_distance:
            yield s, min_dist, min_yard

    def get_outer_target_cell(ship, enemy, defend_yard_dist, defend_yard):
      min_dist = 999
      min_dist_cell = None
      for cell in get_neighbor_cells(enemy.cell):
        dist = manhattan_dist(cell.position, defend_yard.position, self.c.size)
        if dist == defend_yard_dist + 1:
          d = manhattan_dist(ship.position, cell.position, self.c.size)
          if d < min_dist:
            min_dist = d
            min_dist_cell = cell
      assert min_dist_cell
      return min_dist_cell

    def defend_ship_ratio():
      if self.num_ships <= 10:
        ratio = 0.3
      elif self.num_ships <= 12:
        ratio = 0.4
      elif self.num_ships <= 16:
        ratio = 0.5
      else:
        ratio = 0.6
      return ratio

    enemy_ships = list(all_enemy_ships(LOOSE_ENEMY_SHIP_DEFEND_DIST))
    enemy_ships.sort(key=lambda x: (x[1], -x[0].halite))
    ship_budget = min(MAX_DEFEND_SHIPS,
                      int(defend_ship_ratio() * self.num_ships))
    for enemy, enemy_to_defend_yard_dist, defend_yard in enemy_ships:

      def dist_to_enemy(ship):
        return manhattan_dist(ship.position, enemy.position, self.c.size)

      def dist_to_defend_yard(ship):
        return manhattan_dist(ship.position, defend_yard.position, self.c.size)

      if (ship_budget > 0 and
          enemy_to_defend_yard_dist <= TIGHT_ENEMY_SHIP_DEFEND_DIST):
        ships = [
            s for s in self.my_idle_ships
            if (s.halite < enemy.halite and
                dist_to_defend_yard(s) <= enemy_to_defend_yard_dist)
        ]
        ships.sort(key=dist_to_enemy)
        for ship in ships[:min(3, ship_budget)]:
          target_cell = get_outer_target_cell(
              ship, enemy, enemy_to_defend_yard_dist, defend_yard)
          self.assign_task(ship, target_cell, ShipTask.DESTORY_ENEMY_TASK_INNER)
          ship_budget -= 1

      if ship_budget > 0:
        ships = [
            s for s in self.my_idle_ships
            if (s.halite < enemy.halite and
                dist_to_defend_yard(s) >= enemy_to_defend_yard_dist)
        ]
        ships.sort(key=dist_to_enemy)
        for ship in ships[:min(ship_budget, 3)]:
          self.assign_task(ship, enemy.cell, ShipTask.DESTORY_ENEMY_TASK_OUTER)
          ship_budget -= 1

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
    """Builds shipyard with a random ship if we have enough halite and ships."""
    convert_cost = self.board.configuration.convert_cost
    me = self.me

    # No ship left.
    if not me.ship_ids:
      return

    # Keep balance for the number of ships and shipyards.
    num_ships = len(me.ship_ids)
    num_shipyards = len(me.shipyard_ids)
    if num_shipyards * SHIP_TO_SHIYARD_FACTOR > num_ships:
      return

    convert_threshold = MIN_HALITE_TO_BUILD_SHIPYARD
    if num_shipyards == 0 or self.board.step <= BEGINNING_PHRASE_END_STEP:
      convert_threshold = convert_cost

    def select_ship(ship):
      if ship.halite + me.halite < max(convert_cost, convert_threshold):
        return -1

      # It's on a shipyard
      if ship.cell.shipyard_id:
        return -1

      # Put it away from one of the nearest shipyard.
      val = 0
      min_dist, min_yard = self.find_nearest_shipyard(ship, self.me.shipyards)
      if min_yard:
        if min_dist >= 7:
          val += 100
        else:
          return -1

      # Maximize the total number of halite cells when converting ship.
      for cell in self.board.cells.values():
        dist = manhattan_dist(ship.position, cell.position, self.c.size)
        if (dist <= NEARBY_HALITE_CELLS_DIST and cell.halite > 0 and
            cell.position != ship.position):
          val += 1
      return val

    ship_scores = [(select_ship(ship), ship) for ship in me.ships]
    ship_scores.sort(key=lambda x: x[0], reverse=True)
    for score, ship in ship_scores:
      if score < 0:
        continue

      me._halite -= (convert_cost - ship.halite)
      ship.next_action = ShipAction.CONVERT
      ship.has_assignment = True
      ship.cell.is_targetd = True

      # Only build one shipyard at a time.
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

      if ship.task_type == ShipTask.DESTORY_ENEMY_TASK_OUTER:
        enemy = target_cell.ship
        wt += (convert_cost + enemy.halite) / (dist + 1)

        # TODO(wangfei): think again
        _, defend_yard = self.find_nearest_shipyard(enemy, self.me.shipyards)
        if defend_yard:
          dist_to_yard = manhattan_dist(next_position, defend_yard.position,
                                        self.c.size)
          wt += -enemy.halite / (dist_to_yard + 1)

      if ship.task_type == ShipTask.DESTORY_ENEMY_TASK_INNER:
        wt += convert_cost / (dist + 1)

      # If there is an enemy in next_position with lower halite
      for nb_cell in get_neighbor_cells(next_cell, include_self=True):
        if (has_enemy_ship(nb_cell, self.me) and
            nb_cell.ship.halite <= ship.halite):
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

    # PRINT_ME
    # print('#', self.board.step, 'ships=', len(ships),
    # 'halite=', self.me.halite, 'cargo=',
    # sum([s.halite for s in self.me.ships], 0), 'max(enemy halite)=',
    # max(e.halite for e in self.board.opponents))
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
    board = self.board
    me = board.current_player

    def max_ship_num():
      return MAX_SHIP_NUM + max(0, (self.me.halite - 4000) // 2000)

    def is_shipyard_in_danger(yard):
      # If there is one of my ship on yard, it's safe.
      if yard.cell.ship_id and yard.cell.ship.player_id == self.me.id:
        return False
      for nb_cell in get_neighbor_cells(yard.cell):
        if has_enemy_ship(nb_cell, self.me):
          return True
      return False

    def spawn(yard):
      me._halite -= board.configuration.spawn_cost
      yard.next_action = ShipyardAction.SPAWN

    def spawn_threshold():
      if board.step <= BEGINNING_PHRASE_END_STEP:
        return self.c.spawn_cost
      return MIN_HALITE_TO_BUILD_SHIP

    # Spawn ship if yard is in danger.
    shipyards = me.shipyards
    if len(shipyards) <= 1:
      for y in shipyards:
        if is_shipyard_in_danger(y) and me.halite >= self.c.spawn_cost:
          spawn(y)

    # Too many ships.
    num_ships = len(me.ship_ids)
    if num_ships >= max_ship_num():
      return

    if num_ships >= 2 and self.step >= ENDING_PHRASE_STEP:
      return

    random.shuffle(shipyards)
    for shipyard in shipyards:
      # print('shipyard', shipyard.id, 'at', shipyard.position)

      # Skip if not pass threshold when any ship is alive.
      if num_ships and me.halite < spawn_threshold():
        continue

      spawn(shipyard)

  def final_stage_back_to_shipyard(self):
    MARGIN_STEPS = 10
    MIN_HALITE_TO_YARD = 10

    def ship_and_dist_to_yard():
      for ship in self.my_idle_ships:
        if ship.halite <= MIN_HALITE_TO_YARD:
          continue
        _, yard = self.find_nearest_shipyard(ship, self.me.shipyards)
        if yard:
          dist = manhattan_dist(ship.position, yard.position, self.c.size)
          yield dist, ship, yard

    if not self.me.shipyard_ids:
      return

    ship_dists = list(ship_and_dist_to_yard())
    if not ship_dists:
      return

    max_dist = max(max(d for d, _1, _2 in ship_dists), self.num_ships)
    if self.step + max_dist + MARGIN_STEPS < self.c.episode_steps:
      return

    for _, ship, min_dist_yard in ship_dists:
      self.assign_task(ship, min_dist_yard.cell,
                       ShipTask.RETURN_TO_SHIPYARD_TASK)

  initial_shipyard_set = False
  initial_yard_position = None
  initial_ship_position = None

  def convert_first_shipyard(self):
    """Strategy for convert the first ship yard."""
    MAX_MOVE_TO_INITIAL_POSITION = 3
    ESTIMATE_CELL_DIST = 4

    assert self.num_ships == 1, self.num_ships
    ship = self.me.ships[0]

    if not self.initial_ship_position:
      ShipStrategy.initial_ship_position = ship.position

    def estimate_cell(candidate_cell):
      num = 0
      halite = 0
      for cell in self.board.cells.values():
        dist = manhattan_dist(candidate_cell.position, cell.position,
                              self.c.size)
        if (dist <= ESTIMATE_CELL_DIST and
            cell.position != candidate_cell.position):
          num += int(cell.halite > 0)
          halite += cell.halite
      return num, halite, candidate_cell

    def select_initial_cell():
      for cell in self.board.cells.values():
        dist = manhattan_dist(self.initial_ship_position, cell.position,
                              self.c.size)
        if dist <= MAX_MOVE_TO_INITIAL_POSITION:
          yield estimate_cell(cell)

    if not self.initial_yard_position:
      candidate_cells = list(select_initial_cell())
      if candidate_cells:
        candidate_cells.sort(key=lambda x: (-x[0], -x[1]))
        num, halite, yard_cell = candidate_cells[0]
        ShipStrategy.initial_yard_position = yard_cell.position
        # print("Ship initial:", self.initial_ship_position, 'dist=',
        # manhattan_dist(self.initial_ship_position,
        # self.initial_yard_position, self.c.size),
        # 'selected yard position:', self.initial_yard_position, 'num=',
        # num, 'halite=', halite)

    if ship.position == self.initial_yard_position:
      ShipStrategy.initial_shipyard_set = True
      ship.next_action = ShipAction.CONVERT
      ship.has_assignment = True
      ship.cell.is_targetd = True
    else:
      self.assign_task(ship, self.board[self.initial_yard_position],
                       ShipTask.GOTO_INITIAL_YARD_POS_TASK)

  def execute(self):
    self.collect_game_info()

    if self.initial_shipyard_set:
      self.convert_to_shipyard()
      self.spawn_ships()

      if (self.step > BEGINNING_PHRASE_END_STEP or
          len(self.me.ship_ids) >= MAX_SHIP_NUM):
        self.attack_enemy_yard()
        self.attack_enemy_ship()

      self.final_stage_back_to_shipyard()
      self.continue_mine_halite()
      self.send_ship_to_home_yard()
      self.send_ship_to_halite()
    else:
      self.convert_first_shipyard()

    # TODO: add backup mining strategy without limit.
    self.compute_ship_moves()
    self.collision_check()


def agent(obs, config):
  board = Board(obs, config)
  ShipStrategy(board).execute()
  return board.current_player.next_actions
