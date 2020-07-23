# -*- coding: utf-8 -*-

import json
import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

import agent_bee_v4_0_4_1 as agent

work_dir = "/ssd/wangfei/repo/flynn/kaggle_halite4/"
replay_path = os.path.join(work_dir, "debug_replay", "1746731.json")


def agent(name, collection=False):
  return os.path.join(work_dir, collection and 'collection' or '', name)


config = Config(max_depth=10)
graphviz = GraphvizOutput(output_file='filter_max_depth.png')

with open(replay_path, 'r') as f:
  replay_json = json.loads(f.read())

strategy = agent.ShipStrategy()
replayer = Replayer(strategy, replay_json, player_id=1)
replayer.play(250)

with PyCallGraph(output=graphviz, config=config):
  replayer.play(100)
