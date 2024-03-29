# -*- coding: utf-8 -*-

import json
import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput
from kaggle_environments.envs.halite.helpers import *

import agent_bee_v4_16_07 as agent
from replayer import Replayer

# work_dir = "/ssd/wangfei/repo/flynn/kaggle_halite4/"
work_dir = "/home/wangfei/repo/flynn/kaggle_halite4"

replay_path = '/home/wangfei/data/20200801_halite/episodes/collection/3246009.json'
result_path = os.path.join(work_dir, "%s.png" % ("agent_bee_4_16_07.py"))


config = Config(max_depth=10)
graphviz = GraphvizOutput(output_file=result_path)

with open(replay_path, 'r') as f:
  replay_json = json.loads(f.read())

strategy = agent.ShipStrategy()
replayer = Replayer(strategy, replay_json, player_id=1)
replayer.play(250, print_step=True)
with PyCallGraph(output=graphviz, config=config):
  replayer.play(100, print_step=True)
