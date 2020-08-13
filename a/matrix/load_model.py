# from matrix_v0 import get_model

MODEL_PATH = "/home/wangfei/data/20200801_halite/model/unet_L7_v13"

# model = get_model()
# print(MODEL_PATH)
# model.save_weights(MODEL_PATH)
# model.load_weights(MODEL_PATH)

import train

t = train.Trainer(None, MODEL_PATH)
t.on_batch_finished()
