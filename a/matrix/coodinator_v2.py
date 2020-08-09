import argparse
import os
import json
import time
import subprocess
import datetime
import uuid
import random
from multiprocessing import Pool, Queue, Process

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

NUM_PROCESSES = 6
BATCH_SIZE = 6
EPISODE_STEPS = 10

def replay_to_ship_rewards(replay_json):
  import train

  returns = []
  total_deposits = []
  total_collects = []

  num_players = len(replay_json['rewards'])
  for player_id in range(num_players):
    boards = train.gen_player_states(replay_json, player_id)
    boards = list(boards)

    ship_rewards = train.compute_returns(list(boards), as_list=True)
    returns = np.concatenate(list(ship_rewards.values()) + [returns])

    b = boards[-1]
    total_deposits.append(b.total_deposite)
    total_collects.append(b.total_collect)
  return returns, total_deposits, total_collects


def compute_grad(model_dir, replay_queue, grad_queue):
  print("Hi, compute_grad: ready", os.getpid())
  import train

  trainer = train.Trainer(None, model_dir, return_params=(0, 0))
  while True:
    replay_json = replay_queue.get()
    num_players = len(replay_json['rewards'])
    player_ids = list(range(num_players))

    # Sample one
    # pid = random.choice(player_ids)
    # player_ids = [pid]

    player_boards = [
      list(train.gen_player_states(replay_json, player_id))
      for player_id in player_ids
    ]
    grads = trainer.train(player_boards, apply_grad=False)
    grad_queue.put(grads)


def gradient_consumer(model_dir, grad_queue, episode_queue):
  print("Hi, gradient_consumer: ready")

  import train
  trainer = train.Trainer(None, model_dir)

  batch_count = 0
  grads_buffer = []
  while True:
    grads = grad_queue.get()
    grads_buffer.extend(grads)

    print("grad buffer size: ", len(grads_buffer))
    if len(grads_buffer) >= BATCH_SIZE * 4:
      batch_count += 1
      # random.shuffle(grads_buffer)
      for i, g in enumerate(grads_buffer):
        trainer.apply_grad(g)
        print("batch[%s]: apply grad at %s" % (batch_count, i))
      trainer.on_batch_finished()
      grads_buffer.clear()
      episode_queue.put(True)


def start_grad_processes(model_dir, replay_queue, grad_queue):
  grad_processes = []
  for i in range(NUM_PROCESSES):
    p = Process(target=compute_grad, args=(model_dir, replay_queue, grad_queue))
    p.start()
    grad_processes.append(p)
  return grad_processes


def run_simulation(model_dir, simulation_queue, replay_queue):
  print("Hi, simulation ready at ", os.getpid())

  from train import Trainer
  import matrix_v0 as mat
  from kaggle_environments import make

  while True:
    epsilon, batch_dir = simulation_queue.get()

    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_name = '%s_%s.json' % (time_tag, str(uuid.uuid4())[:5])
    output_path = os.path.join(batch_dir, episode_name)

    # prepare model
    trainer = Trainer(mat.get_model(), model_dir)
    mat.STRATEGY = mat.ShipStrategy(trainer.model, epsilon)

    env = make("halite", {'episodeSteps': EPISODE_STEPS}, debug=True)
    env.run([mat.agent] * 4)

    print('Output episode:', output_path)
    replay_json = env.toJSON()
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(json.dumps(replay_json))
    os.rename(tmp_path, output_path)

    replay_queue.put(replay_json)


def start_simulation_processes(model_dir, simulation_queue, replay_queue):
  sim_processors = []
  for i in range(NUM_PROCESSES):
    p = Process(target=run_simulation, args=(model_dir, simulation_queue, replay_queue))
    p.start()
    sim_processors.append(p)
  return sim_processors


def run(episode_dir, model_dir, batch):
  epsilon = 0.1
  epsilon_decay = 0.99

  simulation_queue = Queue()
  replay_queue = Queue()
  grad_queue = Queue()
  episode_queue = Queue()

  grad_consumer = Process(target=gradient_consumer,
                          args=(model_dir, grad_queue, episode_queue))
  grad_consumer.start()

  processers = start_grad_processes(model_dir, replay_queue, grad_queue)

  sim_processor = start_simulation_processes(model_dir, simulation_queue, replay_queue)

  for b in range(batch):
    print("Step: %s, epsilon value: %s" % (b, epsilon))

    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(episode_dir, time_tag)
    if not os.path.exists(batch_dir):
      os.makedirs(batch_dir)
    for i in range(BATCH_SIZE):
      args = (epsilon, batch_dir)
      simulation_queue.put(args)

    # End of one batch.
    episode_queue.get()
    print(F"batch[{batch}]: all episodes finished at this iteration.")

    epsilon *= epsilon_decay


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('-b', '--batch', required=True, type=int)
  args = parser.parse_args()
  run(args.episode_dir, args.model_dir, args.batch)


if __name__ == "__main__":
  main()
