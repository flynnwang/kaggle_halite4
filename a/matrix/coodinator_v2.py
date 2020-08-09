import argparse
import os
import json
import subprocess
import datetime
import uuid
import random
from multiprocessing import Pool, Queue, Process

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BATCH_SIZE = 24
EPISODE_STEPS = 200


def compute_grad(args):
  replay_json, model_dir = args

  print("Hi, compute_grad: ready", os.getpid())
  import train

  num_players = len(replay_json['rewards'])
  player_ids = list(range(num_players))

  # Sample one
  # pid = random.choice(player_ids)
  # player_ids = [pid]

  player_boards = [
    list(train.gen_player_states(replay_json, player_id))
    for player_id in player_ids
  ]
  trainer = train.Trainer(None, model_dir, return_params=(0, 0))
  grads = trainer.train(player_boards, apply_grad=False)
  return grads


def gradient_consumer(model_dir, grad_queue, episode_queue):
  print("Hi, gradient_consumer: ready")

  import train

  def apply_gradients(grads_buffer):
    trainer = train.Trainer(None, model_dir)
    for i, g in enumerate(grads_buffer):
      trainer.apply_grad(g)
      print("batch[%s]: apply grad at %s" % (batch_count, i))
    trainer.on_batch_finished()

  batch_count = 0
  grads_buffer = []
  while True:
    grads = grad_queue.get()
    apply_gradients(grads)
    episode_queue.put(True)


def gradient_pool(model_dir, replay_queue, grad_queue):

  def scan_for_replays(episode_dir):
    for name in os.listdir(episode_dir):
      episode_path = os.path.join(episode_dir, name)
      print("Loading:", episode_path)
      with open(episode_path, 'r') as f:
        replay_json = json.loads(f.read())
      yield replay_json

  def gen_args(replay_jsons):
    for replay_json in replay_jsons:
      yield replay_json, model_dir

  with Pool() as pool:
    while True:
      batch_dir = replay_queue.get()
      print('result from replay_queue: ', batch_dir)
      replay_jsons = list(scan_for_replays(batch_dir))

      all_grads_list = []
      for grads_list in pool.imap_unordered(compute_grad, gen_args(replay_jsons)):
        all_grads_list.extend(grads_list)

      grad_queue.put(all_grads_list)


def run_simulation(model_dir, simulation_queue, replay_queue):
  print("Hi, simulation ready at ", os.getpid())
  from kaggle_environments import make
  while True:
    batch_dir, epsilon  = simulation_queue.get()
    print("Run lsd on: ", batch_dir)
    subprocess.check_call(["python3", "lsd.py", "-o", batch_dir, '-m', model_dir,
                           '--epsilon', str(epsilon), '--episode_steps', str(EPISODE_STEPS),
                           '--batch_size', str(BATCH_SIZE)])
    print("Lsd finished at", batch_dir)
    replay_queue.put(batch_dir)


def run(episode_dir, model_dir, batch, epsilon):
  epsilon_decay = 0.998

  simulation_queue = Queue()
  replay_queue = Queue()
  grad_queue = Queue()
  episode_queue = Queue()

  sim_processor = Process(target=run_simulation,
                          args=(model_dir, simulation_queue, replay_queue))
  sim_processor.start()

  gradient_producer = Process(target=gradient_pool,
                              args=(model_dir, replay_queue, grad_queue))
  gradient_producer.start()

  grad_consumer = Process(target=gradient_consumer,
                          args=(model_dir, grad_queue, episode_queue))
  grad_consumer.start()

  for b in range(batch):
    print("Step: %s, epsilon value: %s" % (b, epsilon))

    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(episode_dir, time_tag)
    simulation_queue.put((batch_dir, epsilon))

    # End of one batch.
    episode_queue.get()
    print(F"batch[{b}]: all episodes finished at this iteration.")

    epsilon *= epsilon_decay


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode_dir', required=True)
  parser.add_argument('-m', '--model_dir', required=True)
  parser.add_argument('-b', '--batch', required=True, type=int)
  parser.add_argument('--epsilon', type=float, default=1.0)
  parser.add_argument('--episode_steps', type=int, default=400)
  parser.add_argument('--batch_size', type=int, default=24)
  args = parser.parse_args()

  global EPISODE_STEPS, BATCH_SIZE
  EPISODE_STEPS = args.episode_steps
  BATCH_SIZE = args.batch_size
  run(args.episode_dir, args.model_dir, args.batch, args.epsilon)


if __name__ == "__main__":
  main()
