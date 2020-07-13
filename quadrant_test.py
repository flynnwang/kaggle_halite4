from kaggle_environments.envs.halite.helpers import Point


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


def test_get_qudrant():
  assert get_quadrant(Point(0, 0)) == 0

  assert get_quadrant(Point(1, 0)) == 1
  assert get_quadrant(Point(1, 1)) == 1

  assert get_quadrant(Point(0, 1)) == 2
  assert get_quadrant(Point(-1, 1)) == 2

  assert get_quadrant(Point(-1, 0)) == 3
  assert get_quadrant(Point(-1, -1)) == 3

  assert get_quadrant(Point(0, -1)) == 4
  assert get_quadrant(Point(1, -1)) == 4
