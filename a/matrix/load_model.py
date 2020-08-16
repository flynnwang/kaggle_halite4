# from matrix_v0 import get_model

import getpass

USER = getpass.getuser()

MODEL_PATH = "/home/%s/data/20200801_halite/model/unet_test9x9_v9" % (USER)


# model = get_model()
# print(MODEL_PATH)
# model.save_weights(MODEL_PATH)
# model.load_weights(MODEL_PATH)

import train

t = train.Trainer(None, MODEL_PATH)
t.on_batch_finished()
