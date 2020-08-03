
import os
import argparse
from multiprocessing import Pool
import numpy as np

BATCH_SIZE = 25
EPISODE_STEPS = 60

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def simulate(v):

  import tensorflow as tf
  print(v)
  return tf.convert_to_tensor(np.ones((10, )))


def run_lsd():

  def gen_simulations():
    v = list(range(10))
    with Pool(processes=3) as pool:
      values = []
      for t in pool.imap_unordered(simulate, v):
        values.append(v)
      print(values)

  gen_simulations()


def main():
  parser = argparse.ArgumentParser()
  run_lsd()


if __name__ == "__main__":
  main()
