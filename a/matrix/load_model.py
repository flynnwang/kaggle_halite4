from matrix_v0 import get_model, MODEL_PATH


model = get_model()
print(MODEL_PATH)
model.save_weights(MODEL_PATH)
model.load_weights(MODEL_PATH)



import time

time.sleep(300)
