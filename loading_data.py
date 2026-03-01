import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

cosine_sim = np.load("data/data.npy")
matrix = np.load("data/songs.npy")

df = pd.read_csv("data/spotify_millsongdata.csv")


def load():
    return cosine_sim, matrix, df[:20000]
