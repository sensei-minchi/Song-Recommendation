import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_data():
    file_path = "spotify_millsongdata.csv"
    
    try:
        with torch.serialization.safe_globals([ 
        'pandas.core.series.Series',  # allow this global
        ]):
            song_embeddings = torch.load("data/song_embeddings.pt", weights_only=False)
        
        df = pd.read_csv("data/spotify_millsongdata.csv")
        song_embeddings_np = np.array([t.detach().cpu().numpy() for t in song_embeddings])
        
    except FileNotFoundError:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "notshrirang/spotify-million-song-dataset",
            file_path,
        )
        
        df.drop('link', inplace=True, axis=1)
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model = SentenceTransformer(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        def get_song_embedding(lyrics):
            embedding = model.encode(
                lyrics,
                convert_to_tensor=True,
                normalize_embeddings=True  
            )
            return embedding
        
        df["embeddings"] = df["text"].apply(get_song_embedding)
        torch.save(df["embeddings"], "data/song_embeddings.pt")

        song_embeddings_np = np.array([t.detach().cpu().numpy() for t in df["embeddings"]])

    return song_embeddings_np, df
