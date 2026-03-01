from tensorflow.keras.models import load_model
from loading_data import load

cosine_sim, matrix, df = load()
user = load_model("user_model.keras")


def predict(song_name, model=user, df=df, matrix=matrix):
    idx_list = df[df["song"].str.lower() == song_name.lower()].index
    idx = idx_list[0]

    song_vector = matrix[idx].reshape(1, -1)

    rate_pred = model.predict(song_vector)

    return rate_pred[0][0]


song_name = input("Enter song:")
print("Chances of user liking the song are: {}".format(predict(song_name, user)))
