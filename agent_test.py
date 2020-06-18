from agent0 import *


def test_manhatten_dist():
  size = 21

  a = Point(0, 0)
  b = Point(0, 5)
  assert manhattan_dist(a, b, size) == 5

  a = Point(0, 0)
  b = Point(5, 0)
  assert manhattan_dist(a, b, size) == 5

  b = Point(5, 0)
  assert manhattan_dist(a, b, size) == 5

  b = Point(0, 20)
  assert manhattan_dist(a, b, size) == 1

  b = Point(20, 0)
  assert manhattan_dist(a, b, size) == 1


def test_compute_next_move():
  a = Point(5, 2)
  assert compute_next_move(a, a) == None

  b = Point(3, 3)
  assert compute_next_move(a, b) == Point(-1, 0)

  b = Point(6, 2)
  assert compute_next_move(a, b) == Point(1, 0)
  b = Point(5, 3)
  assert compute_next_move(a, b) == Point(0, 1)

  b = Point(4, 2)
  assert compute_next_move(a, b) == Point(-1, 0)
  b = Point(5, 1)
  assert compute_next_move(a, b) == Point(0, -1)

  b = Point(20, 2)
  assert compute_next_move(a, b) == Point(+1, 0)


def test_compuet_next_moves():
  a = Point(5, 2)
  b = Point(3, 3)
  assert compute_next_moves(a, b) == [Point(-1, 0), Point(0, +1)]
