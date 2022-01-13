import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import json

st.write("""
# Song suggestion app
This app suggest songs based on your song properties choice!
Data obtained from the [song popularity dataset](https://www.kaggle.com/yasserh/song-popularity-dataset/version/1).
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Choose song features on scale from 0 to 1
""")

with open('columns.json') as f:
    fit_cols = json.load(f)

song_names = pd.read_csv(
    "song_names.csv", sep=",")

# Collects user input features into dataframe
def user_input_features():
    data = {col: st.sidebar.slider(col, 0.0,1.0,0.5) for col in fit_cols}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


load_nn = pickle.load(open('song_nn.pkl', 'rb'))
prediction = load_nn.kneighbors(input_df)

st.subheader('Prediction - showing closest predictions')
st.write(pd.concat([song_names.loc[prediction[1][0]].reset_index(drop=True),
                   pd.DataFrame(prediction[0][0], columns=["song distance"])],
                  axis=1
                  ).style.hide_index(
    ).to_html(), unsafe_allow_html=True)
