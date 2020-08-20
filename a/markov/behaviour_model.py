


import keras
from keras import layers
from kaggle_environments.envs.halite.helpers import *

from model_input import MODEL_INPUT_SIZE, NUM_LAYERS


SHIP_ACTIONS = [a for a in list(ShipAction) if a != ShipAction.CONVERT] + [None]
NUM_SHIP_ACTIONS = len(SHIP_ACTIONS)
INPUT_SHAPE = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, NUM_LAYERS)

def get_keras_unet():
  inputs = layers.Input(INPUT_SHAPE)

  ### [First half of the network: downsampling inputs] ###
  # Entry block
  x = layers.Conv2D(32, 1, strides=1, padding="same",
                    kernel_initializer='he_normal')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [32, 64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer='he_normal')(
      previous_block_activation
    )
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual


  flattened_x = layers.Flatten()(x)
  x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(flattened_x)
  x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
  ship_action_output = layers.Dense(NUM_SHIP_ACTIONS, activation='softmax')(x)

  model = keras.Model(inputs, outputs=ship_action_output)
  return model
