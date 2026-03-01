## Song Recommendation System v2.0
An advanced collaborative recommendation system that treats a user's listening history as a temporal sequence, using LSTM layers and High-Dimensional Lyric Embeddings to capture the shifting "vibe" and emotional context of a user's journey.

### New Updates
1) Instead of using TF-IDF vectorizer for converting lyrics to vector, sentence transformer is being used which captures context from the lyrics
2) Using LSTM to capture the temporal sequence (transfomer also could have been used but to save memory and computational time I didn't)

### The Vision
Most recommenders look at what you bought/listened to. This system looks at the essence of what you listened to. By converting lyrics into normalized embeddings, the model understands the semantic meaning behind the music, allowing it to recommend songs that fit the "mood sequence" of a session.

### Tech Stack
Deep Learning: TensorFlow/Keras (LSTM architecture)
NLP: Sentence-Transformers (all-mpnet-base-v2) for 768-dim lyric embeddings
Data: Spotify Million Dataset

### How is it different from standard recommendation systems
1) *Using LSTM*: LSTM helps in capturing the "current mood" of the user. As a user listens to a sequence of songs, the LSTM maintains an internal Hidden State.
2) *Updated Song rating function*: In generic collaborative filtering systems, a user vector u is generated and to get the likelihood of the user liking that song, dot product or cosine similarity of these two vectors are calculated, more the score, more the likelihood of the user liking that song. In this system, a neural network will be maintained for each user and as neural network is just a fancy and complex math function, the neural network will predict the likelihood of the song being liked by the user
   

### Pending updates
1) When deployed for large scale applications, it may get difficult to store neural network file for each user, so to **reduce the memory consumption**, we will prune and quantize the neural network, this will reduce the memory consumption by upto 20-30%

2) **Adding song context window**: To recommend new songs, the next version will give more importance to the the songs that have been heard recently. The plan is to create windows and add positional embedding (most probably RoPE) to recommend new songs

3) Adding more features like genre, danceability, time etc to make better recommendations for the user
