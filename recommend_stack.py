import numpy as np
import pandas as pd
from loading_data import load

cosine_sim, matrix, df = load()


def add_by_artist(artist, matrix=matrix, df=df):
    idx_list = df[df["artist"].str.lower() == artist].index

    stack = matrix[idx_list]
