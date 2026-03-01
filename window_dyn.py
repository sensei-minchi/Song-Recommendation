from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Input
from loading_data import load_data
import numpy as np

def format_sequences(user_history_embeddings, user_history_rating, window_size=10):
    X = []
    y = []
    
    for i in range(len(user_history_embeddings) - window_size):
        window = user_history_embeddings[i : i + window_size]
        window_rating = user_history_rating[i : i + window_size]
        X.append(window)
        y.append(user_history_rating[i + window_size])  # Use the last rating in the window as target
        
    return np.array(X), np.array(y)


def train(x, y, epochs=5):
    x,y = format_sequences(x, y)
    window_size = 10 # Let's look at the last 10 songs

    model = Sequential([
        # No Reshape(1, 768) needed if the input is already a sequence
        layers.Input(shape=(window_size, 768)), 
        layers.LSTM(256),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
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
    user.save("user_model_temporal.keras")

