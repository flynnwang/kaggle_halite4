from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *


def parse_board(replay_json, player_id, conf, step):
  state = replay_json['steps'][step][0]
  obs = state['observation']
  obs['player'] = player_id
  return Board(obs, conf)

class Replayer:

  def __init__(self, strategy, replay_json, player_id=0):
    self.strategy = strategy
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = make("halite",
                    configuration=replay_json['configuration'],
                    steps=replay_json['steps'])
    self.step = 0
    num_state = len(self.replay_json['steps'])
    self.boards = [parse_board(replay_json, player_id, self.env.configuration, s)
                   for s in range(num_state)]

  def play(self, steps=1, print_step=False):
    for i in range(steps):
      if print_step:
        print("Step %s" % self.step)

      self.simulate(i)
      self.step += 1

  def simulate(self, step):
    # list of length 1 for each step
    board = self.boards[step]
    self.strategy.update(board)
    self.strategy.execute()
