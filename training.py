from user_model import train
from loading_data import load
import numpy as np

cosine_sim, matrix, df = load()

idx1 = df[df["artist"].str.lower() == "Foo fighters"].index
idx3 = df[df["artist"].str.lower() == "green day"].index
idx2 = df[df["artist"].str.lower() == "alabama"].index

idx = np.concatenate((idx1, idx2, idx3), axis=0)
X = np.array(matrix[idx])
y = np.array(len(idx1) * [1] + len(idx2) * [0] + len(idx3) * [1]).reshape(-1, 1)

user = train(X, y, epoch=10)

user.save("user_model.keras")

