# -*- coding: utf-8 -*-

import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

work_dir = "/ssd/wangfei/repo/flynn/kaggle_halite4/"
replay_path = os.path.join(work_dir, "debug_replay", "1746731.json")


def agent(name, collection=False):
  return os.path.join(work_dir, collection and 'collection' or '', name)


def run():
  Replayer()
  env = make("halite", {'episodeSteps': 100}, debug=True)
  a = agent("agent_bee_v2.1.1.py")
  env.run([a, "random", "random", "random"])


config = Config(max_depth=10)
graphviz = GraphvizOutput(output_file='filter_max_depth.png')

with PyCallGraph(output=graphviz, config=config):
  run()
