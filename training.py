from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Input
from loading_data import load_data
import numpy as np

def train(x, y, epochs=5):
    model = Sequential([
        Input(shape=(768,)),
        Reshape((1, 768)),
        LSTM(256),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x, y, epochs=epochs)
    return model


if __name__ == "__main__":
    song_embeddings_np, df = load_data()

    # Synthetic user made
    rock_artists = ["Queen", "Nirvana", "The Beatles", "Foo Fighters"]
    jazz_artists = ["Miles Davis", "John Coltrane", "Herbie Hancock", "Louis Armstrong"]

    arr_rock = song_embeddings_np[df["artist"].isin(rock_artists)]
    arr_jazz = song_embeddings_np[df["artist"].isin(jazz_artists)]

    song_hist = np.concatenate((arr_jazz, arr_rock))
    labels = np.concatenate((np.zeros(len(arr_jazz)), np.ones(len(arr_rock))))

    x = song_hist
    y = labels

    user = train(x, y, epochs=5)
    user.save("user_model.keras")

