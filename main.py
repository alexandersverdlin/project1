import pandas as pd
import numpy as np
import requests
from selenium.webdriver import Chrome
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit as st


st.title('Ежжи ежжи эзжиэ')
st.markdown('''aaaaaaaaaaaaaahjhjhbjhb''')


with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # df_full = pd.read_csv("C:/Users/asverdlin/Downloads/tracks.csv")
    # df_lite = df_full.sample(frac=1)[0:100000]
    # df_lite.to_csv('C:/Users/asverdlin/Downloads/track_lite.csv')

    df = pd.read_csv("tracks_lite.csv")

    df.sort_values(by='popularity', ascending=False)[0:1000]

    regr = LinearRegression()

    columns = [
        'duration_ms',
        'explicit',
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'tempo']

    df_coefs = pd.DataFrame()

    for column in columns:
        X = df[[column]]
        Y = df['popularity']
        regr.fit(X, Y)
        plt.plot(1)
        df.plot.scatter(column, 'popularity', alpha=0.1)
        plt.plot(X[column], regr.predict(X), color='C1')
        fig = plt.plot()
        st.pyplot(fig)
        df_coefs[column] = regr.coef_

    coefs = df_coefs.transpose().sort_values(0).rename(columns={0: "coefficient"})
    plt.plot(1)
    coefs.plot.bar(color = 'orange')
    fig = plt.plot()
    st.pyplot(fig)