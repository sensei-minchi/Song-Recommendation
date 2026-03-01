from tensorflow.keras.models import load_model
from loading_data import load_data
import numpy as np

song_embeddings_np, df = load_data()
user = load_model("user_model.keras")


def predict(song_name, model=user, df=df, embeddings=song_embeddings_np):
    idx_list = df[df["song"].str.lower() == song_name.lower()].index
    if len(idx_list) == 0:
        return None
    idx = idx_list[0]

    song_vector = embeddings[idx].reshape(1, -1)

    rate_pred = model.predict(song_vector)

    return rate_pred[0][0]


if __name__ == "__main__":
    song_name = input("Enter song: ")
    result = predict(song_name, user)
    if result is not None:
        print("Chances of user liking the song are: {}".format(result))
    else:
        print("Song not found in dataset.")
