from kaggle_environments import evaluate, make
import os

work_dir = "/ssd/wangfei/repo/flynn/kaggle_halite4/"


def agent(name, collection=False):
  return os.path.join(work_dir, collection and 'collection' or '', name)


def run():
  env = make("halite", {'episodeSteps': 100}, debug=True)
  a = agent("agent_bee_v2.1.1.py")
  env.run([a, "random", "random", "random"])


from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

config = Config(max_depth=10)
graphviz = GraphvizOutput(output_file='filter_max_depth.png')

with PyCallGraph(output=graphviz, config=config):
  run()
