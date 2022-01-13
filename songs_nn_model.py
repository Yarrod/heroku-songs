import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import pandas as pd
import json
songs = pd.read_csv(
    "song_data.csv.gz", sep=",", compression="gzip")

# Ordinal feature encoding
# https://www.kaggle.com/yasserh/song-popularity-dataset/version/1
df = songs.copy()
min_max_scaler = preprocessing.MinMaxScaler()


fit_cols = ['acousticness',
            'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
            'loudness', 'tempo']

neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(min_max_scaler.fit_transform(df[fit_cols]))

# Saving the model
pickle.dump(neigh, open('song_nn.pkl', 'wb'))
df[["song_name","song_popularity"]].to_csv("song_names.csv",index=False)

# saving selections 
with open('columns.json', 'w') as f:
    json.dump(fit_cols, f)
