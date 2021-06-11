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


st.title('Как написать трек, который разорвет чарты: изучаем на данных')
st.subheader('''В первой части с помощью R мы посмотрим на факторы 
            популярности треков, во второй с помощью Python построим 
            регрессии, граф фитов популярных исполнителей и поймем, 
            насколько музыка - хороший выбор, чтобы стать знаменитым''')

st.markdown('''Мы будем использовать датасет от Spotify - tracks.csv. 
            К сожалению, файл слишком большой, чтобы влезть на GitHub, поэтому нам
            ''')

st.markdown('''К сожалению, файл слишком большой, чтобы влезть на GitHub, поэтому 
            для регрессии нам придется оставить только его пятую часть, рандомные 
            100000 строк. Это сделано с помощью следующего кода:
            ''')

st.code('''
    df_full = pd.read_csv("tracks.csv")
    df_lite = df_full.sample(frac=1)[0:100000]
    df_lite.to_csv('track_lite.csv')
    ''')


with st.echo(code_location="above"):

    df = pd.read_csv("tracks_lite.csv")

    df.sort_values(by='popularity', ascending=False)[0:20]

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

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    for column in columns:
        X = df[[column]]
        Y = df['popularity']
        regr.fit(X, Y)
        plt.plot(1)
        df.plot.scatter(column, 'popularity', alpha=0.1, color='#33CCCC')
        plt.plot(X[column], regr.predict(X), color='#006666')
        fig = plt.plot()
        st.pyplot(fig)
        df_coefs[column] = regr.coef_

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    coefs = df_coefs.transpose().sort_values(0).rename(columns={0: "coefficient"})
    plt.plot(1)
    coefs.plot.bar(color = '#33CCCC')
    fig = plt.plot()
    st.pyplot(fig)

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # df_full = pd.read_csv("tracks.csv")
    # df_feats = df_full.sort_values(by = 'popularity', ascending = False)['artists']
    # df_feats.to_csv('df_feats.csv')

    # artists_full = pd.read_csv("artists.csv")
    # artists_lite = artists_full.sort_values(by = 'followers', ascending = False)[0:1000]
    # artists_lite.to_csv('artists_lite.csv')

    artists = pd.read_csv("artists_lite.csv")
    df_feats = pd.read_csv("df_feats.csv")

    most_followed_artists = artists.sort_values(by='followers', ascending=False)[0:50]

    vertices = list(most_followed_artists['name'])
    vertices

with st.echo(code_location="above"):

    feats = set()
    for star in vertices:
        for song in df_feats['artists']:
            if star in song:
                feats.add(song)

    true_feats = set()
    for feat in feats:
        for star in vertices:
            for star2 in vertices:
                if star != star2:
                    if star in feat and star2 in feat:
                        true_feats.add(feat)

with st.echo(code_location="above"):

    feats_list = [feat.replace("'", "").replace('[', '').replace(']', '').split(', ') for feat in true_feats]

    for feat in feats_list:
        for artist in feat:
            if artist not in vertices:
                feat.remove(artist)

    for el in feats_list:
        if len(el) > 2 or len(el) < 2:
            feats_list.remove(el)

    for el in feats_list:
        if len(el) > 3:
            feats_list.remove(el)

with st.echo(code_location="above"):

    feats_list_tuples = [tuple(el) for el in feats_list]
    feats_list_tuples

with st.echo(code_location="above"):

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(feats_list_tuples)
    net = Network(width='1000px', notebook=True)
    net.from_nx(G)
    net.show("feats.html")