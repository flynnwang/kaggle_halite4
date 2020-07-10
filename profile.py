from kaggle_environments import evaluate, make

work_dir = "/ssd/wangfei/repo/flynn/kaggle_halite4/"


def agent(name, collection=False):
  return os.path.join(work_dir, collection and 'collection' or '', name)


def run():
  env = make("halite", {'episodeSteps': 4}, debug=True)
  a = agent("agent_bee_v2.1.py")
  env.run([a, "random", "random", "random"])
