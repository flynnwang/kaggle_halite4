from agent_hungarian import *


def test_mining_steps():
  assert mining_steps(500, 0.25) == 10
