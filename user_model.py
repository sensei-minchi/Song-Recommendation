from tensorflow.keras.losses import MSE, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

from loading_data import load
import numpy as np

print("Imported successfully")


user = Sequential()
user.add(Dense(512, input_shape=(5000,), activation="relu"))
user.add(Dense(128, activation="relu"))
user.add(Dense(32, activation="relu"))
user.add(Dense(1, activation="sigmoid"))

user.compile(loss=BinaryCrossentropy(), optimizer=Adam())

cosine_sim, matrix, df = load()

def train(X, y, epoch=10):
    user.fit(X, y, epochs=epoch)
    return user



