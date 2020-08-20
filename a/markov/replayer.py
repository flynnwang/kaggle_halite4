# -*- coding: utf-8 -*-

import os
import random
from itertools import tee

import json
import numpy as np
from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

class Replayer:

  def __init__(self, replay_json, player_id=0):
    self.replay_json = replay_json
    self.player_id = player_id
    self.env = make("halite",
                    configuration=replay_json['configuration'],
                    steps=replay_json['steps'])
    self.step = 0
    self.total_steps = len(replay_json['steps'])

  def get_board(self, step, board_cls=Board):
    state = self.replay_json['steps'][step][0]
    obs = state['observation']
    obs['player'] = self.player_id
    num_players = len(self.replay_json['rewards'])

    actions = None
    if step + 1 < self.total_steps:
      actions = [
          self.replay_json['steps'][step + 1][p]['action'] for p in range(num_players)
      ]
    board = board_cls(obs, self.env.configuration, actions)
    board.replay_id = self.replay_json['id']
    return board

  def simulate(self, strategy, step=0):
    board = self.get_board(step)
    strategy.update(board)
    strategy.execute()
    self.step += 1


def gen_player_states(replay_json, player_id, debug=False):
  steps = len(replay_json['steps'])
  replayer = Replayer(None, replay_json, player_id)
  # start from 1 because there ase only 399 actions.
  prev = None
  for i in range(0, steps):
    board = replayer.get_board(i)
    board.step_reward = 0
    if prev:
      board.total_deposite = prev.total_deposite
      board.total_collect = prev.total_collect

    board.debug = debug
    #         print('Step #', board.step)
    board.next()
    prev = board
    # print('S=%s, board.total_deposite=' % (i), board.total_deposite)
    yield board

    if is_player_eliminated(board.current_player,
                            board.configuration.spawn_cost):
      break
  board.step_reward += get_last_step_reward(board)
