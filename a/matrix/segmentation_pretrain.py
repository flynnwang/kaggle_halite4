#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import sys

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

import seaborn as sns

# plt.rcParams['figure.figsize'] = (15.0, 6.0)
# pd.set_option('display.float_format', lambda x: '%.8f' % x)

pd.__version__, np.__version__, sns.__version__

# In[12]:

import glob
import json
import random

# In[3]:

# REPLAY_PATH = "/home/wangfei/Downloads/768202_1350724_bundle_archive"
REPLAY_PATH = "/home/wangfei/Downloads/768202_1366553_bundle_archive"

files = glob.glob(REPLAY_PATH + '/**/*.json', recursive=True)
files = [f for f in files if not '_info' in f]
len(files)

# # Model Inupt Example

# In[18]:

sys.path.insert(0, "/home/wangfei/repo/flynn/kaggle_halite4/a/matrix")

from train import *
from matrix_v0 import *

import tensorflow as tf

from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *

# In[17]:

example_path = random.sample(files, 1)[0]

with open(example_path, 'r') as f:
  replay_json = json.loads(f.read())

# env = make('halite', configuration=replay_json['configuration'], steps=replay_json['steps'])
# env.render(mode="ipython", width=800, height=600)

# In[23]:

PLAYER_ID = player_id = 0
total_steps = len(replay_json['steps'])

g = gen_player_states(replay_json, player_id, debug=False)
boards = list(g)

# In[37]:


def show_images(images, cols=1, titles=None):
  assert ((titles is None) or (len(images) == len(titles)))
  n_images = len(images)
  if titles is None:
    titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
  fig = plt.figure(figsize=(3, 3))
  for n, (image, title) in enumerate(zip(images, titles)):
    a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
    plt.imshow(np.rot90(image))  # Rotate to get the right view with Replay
    a.set_title(title)
  fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
  plt.show()


model_input = ModelInput(boards[27])
maps = model_input.get_input(move_axis=False)
# show_images(maps,
# titles=['Halites', "My ships", "My shipyard", "Enemy ships", "Enemy shipyards"])

# In[36]:

maps.shape

# # Prepare Batch Of Data

# In[42]:

input = model_input.get_input()
input.shape

# In[138]:

from enum import Enum, auto

# class CellCategory(Enum):

#     EMPTY_CELL = auto()   # cell without halite
#     HALITE_CELL = auto()  # halite cell without anyone on it

#     ENEMY_CELL = auto()           # enemy ship cell
#     ENEMY_HALITE_CELL = auto()    # enemy on halite cell
#     ENEMY_SHIPYARD_CELL = auto()  # enemy's shipyard
#     ENEMY_ON_SHIPYARD_CELL = auto()

#     SHIP_CELL = auto()              # my ship on empty cell
#     SHIP_HALITE_CELL = auto()       # my ship on halite
#     SHIPYARD_CELL = auto()          # my shipyard cell
#     SHIP_ON_SHIPYARD_CELL = auto()  # my ship on shipyard


class CellCategory(Enum):

  EMPTY_CELL = auto()  # cell without halite
  HALITE_CELL = auto()  # halite cell without anyone on it

  ENEMY_CELL = auto()  # enemy ship cell
  ENEMY_HALITE_CELL = auto()  # enemy on halite cell
  ENEMY_SHIPYARD_CELL = auto()  # enemy's shipyard
  ENEMY_ON_SHIPYARD_CELL = auto()

  SHIP_CELL = auto()  # my ship on empty cell
  SHIP_HALITE_CELL = auto()  # my ship on halite
  SHIPYARD_CELL = auto()  # my shipyard cell
  SHIP_ON_SHIPYARD_CELL = auto()  # my ship on shipyard


list(CellCategory)

# In[172]:

N_CLASSES = len(CellCategory)


def inupt_to_target(x, as_label=True):
  if as_label:
    y = np.zeros((BOARD_SIZE, BOARD_SIZE, 1), dtype="uint8")
  else:
    y = np.zeros((BOARD_SIZE, BOARD_SIZE, N_CLASSES), dtype="uint8")

#     y = np.zeros((32, 32, 1), dtype="uint8")
  for i in range(BOARD_SIZE):
    for j in range(BOARD_SIZE):
      halite, ship, yard, enemy, enemy_yard = x[i, j, :]
      c = None

      if np.allclose(x[i, j, :], 0):
        c = CellCategory.EMPTY_CELL
      elif halite > 0 and np.allclose(x[i, j, 1:], 0):
        c = CellCategory.HALITE_CELL

      elif halite == 0 and ship > 0 and yard == 0:
        c = CellCategory.SHIP_CELL
      elif halite > 0 and ship > 0 and yard == 0:
        c = CellCategory.SHIP_HALITE_CELL
      elif halite == 0 and ship == 0 and yard > 0:
        c = CellCategory.SHIPYARD_CELL
      elif halite == 0 and ship > 0 and yard > 0:
        c = CellCategory.SHIP_ON_SHIPYARD_CELL

      elif halite == 0 and enemy > 0 and enemy_yard == 0:
        c = CellCategory.ENEMY_CELL
      elif halite > 0 and enemy > 0 and enemy_yard == 0:
        c = CellCategory.ENEMY_HALITE_CELL
      elif halite == 0 and enemy == 0 and enemy_yard > 0:
        c = CellCategory.ENEMY_SHIPYARD_CELL
      elif halite == 0 and enemy > 0 and enemy_yard > 0:
        c = CellCategory.ENEMY_ON_SHIPYARD_CELL
      else:
        assert False

      ## Offset action by 1
      if as_label:
        y[i, j] = c.value - 1
      else:
        y[i, j, c.value - 1] = 1


#             y[i + 5, j+ 5] = c.value
  return y

y = inupt_to_target(input)
y.shape

# In[173]:

# plt.imshow(np.rot90(y[:, :, 0]));

# In[142]:

from tensorflow import keras
import numpy as np

INPUT_SHAPE = input.shape
N_SAMPLE = 75


class ReplayerModelInput(keras.utils.Sequence):
  """Helper to iterate over the data (as Numpy arrays)."""

  def __init__(self, batch_size, replay_paths, cache=True):
    self.batch_size = batch_size
    self.replay_paths = replay_paths

  def __len__(self):
    return len(self.replay_paths)

  def get_input_target(self, path):
    with open(path, 'r') as f:
      replay_json = json.loads(f.read())

    player_id = random.choice(list(range(len(replay_json['rewards']))))

    total_steps = len(replay_json['steps'])
    n = min(N_SAMPLE, total_steps)
    steps = random.sample(list(range(total_steps)), n)

    replayer = Replayer(None, replay_json, player_id)

    X = np.zeros((n,) + INPUT_SHAPE, dtype="float32")
    Y = np.zeros((n, BOARD_SIZE, BOARD_SIZE, 1), dtype="uint8")
    for j, step in enumerate(steps):
      board = replayer.get_board(step)
      model_input = ModelInput(board)

      x = model_input.get_input()
      y = inupt_to_target(x)
      X[j, :] = x
      Y[j, :] = y
    return X, Y

  def __getitem__(self, idx):
    """Returns tuple (input, target) correspond to batch #idx."""
    replay_path = self.replay_paths[idx]
    return self.get_input_target(replay_path)


def extract_one(file_path):
  r = ReplayerModelInput(None, None)
  return r.get_input_target(file_path)


def extract_data(files):
  from multiprocessing import Pool
  inputs = []
  targets = []
  with Pool(processes=6) as pool:
    for i, (input, target) in enumerate(pool.imap_unordered(extract_one,
                                                            files)):
      inputs.append(input)
      targets.append(target)
      print('i=%s / %s' % (i, len(files)))
  X, Y = np.concatenate(inputs), np.concatenate(targets)
  print('X.shape: ', X.shape)
  print('Y.shape: ', Y.shape)
  return X, Y


KERNEL_SIZE = 1


# TODO(wangfei): try use conv with filter size of 1
def get_model(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)

  input_padding = ((5, 6), (5, 6))
  x = layers.ZeroPadding2D(input_padding)(inputs)

  ### [First half of the network: downsampling inputs] ###

  # Entry block
  x = layers.Conv2D(32, KERNEL_SIZE, strides=1, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(filters, 1, strides=2,
                             padding="same")(previous_block_activation)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  ### [Second half of the network: upsampling inputs] ###

  previous_block_activation = x  # Set aside residual

  for filters in [256, 128, 64]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)

    # Project residual
    residual = layers.UpSampling2D(2)(previous_block_activation)
    residual = layers.Conv2D(filters, 1, padding="same")(residual)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  x = layers.Cropping2D(input_padding)(x)
  # Add a per-pixel classification layer
  outputs = layers.Conv2D(num_classes,
                          KERNEL_SIZE,
                          activation="softmax",
                          padding="same")(x)

  # Define the model
  model = keras.Model(inputs, outputs)
  return model


def get_unet_model(input_shape, num_classes, input_padding=((5, 6), (5, 6))):
  # Not converged.
  inputs = Input(shape=input_shape)

  x = layers.ZeroPadding2D(input_padding)(inputs)
  x = layers.Conv2D(32, 1, strides=1, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  conv1 = Conv2D(64,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(x)
  conv1 = Conv2D(64,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv1)
  conv1 = layers.BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(128,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv2)
  conv2 = layers.BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(256,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv3)
  conv3 = layers.BatchNormalization()(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  def decoder(input_tensor):
    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2D(256,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(
                     UpSampling2D(size=(2, 2))(input_tensor))
    print('input_shape', input_tensor.shape)
    print('conv3', conv3.shape)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    return layers.Cropping2D(input_padding)(conv9)

  outputs = layers.Conv2D(num_classes, 1, activation="softmax",
                          padding="same")(decoder(pool3))
  model = keras.Model(inputs, outputs)
  return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
# model = get_model(INPUT_SHAPE, N_CLASSES)
model = get_unet_model(INPUT_SHAPE, N_CLASSES)
model.summary()

# # Training

# In[205]:

batch_size = 32

# In[206]:

import random

# Split our img paths into a training and a validation set
random.Random(42).shuffle(files)

# files = files[:200]
val_samples = 500
train_data = extract_data(files[:-val_samples])
valid_data = extract_data(files[-val_samples:])

# Instantiate data Sequences for each split
# train_gen = ReplayerModelInput(batch_size, train_file_paths)
# val_gen = ReplayerModelInput(batch_size, valid_file_paths)

# In[ ]:

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

MODEL_PATH = "/home/wangfei/data/20200801_halite/model/seg_model/unet_segmentation_v2.h5"
callbacks = [keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)]

# Train the model, doing validation at the end of each epoch.
epochs = 50
model.fit(*train_data,
          epochs=epochs,
          validation_data=valid_data,
          callbacks=callbacks)

pred_true = inupt_to_target(input, as_label=False)
pred = model.predict(np.expand_dims(input, 0))[0, :]
pred.shape, pred_true.shape
