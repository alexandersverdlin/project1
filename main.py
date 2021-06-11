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
st.header('О проекте и используемых данных')
st.subheader('Датасет')
st.markdown('''Мы будем использовать датасет от Spotify. В нем для всех песен на Spotify указана популярность на апрель 2021 года и множество факторов от 0 до 1, которые автоматические рассчитывает Spotify: в том числе energy, danceability, acousticness и т.д. Скоро мы посмотрим на него поближе, но при желании подробнее про датасет можно прочитать тут: https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=artists.csv''')
st.subheader('Описание проекта')
st.markdown('''Проект состоит из двух частей - R и Python. В первой части с помощью R мы поближе посмотрим на факторы популярности треков и сделаем несколько интересных визуализаций. 
Во второй с помощью Python построим регрессии для этих факторов и поймем, каким должен быть трек, чтобы с большей вероятностью стать популярным. Также мы построим граф фитов 50 самых популярных исполнителей на Spotify. А в конце спарсим таблицу самых популярных аккаунтов в Instagram из Википедии, чтобы понять, насколько музыканты популярны относительно других знаменитостей''')
st.subheader('Используемые технологии')
st.markdown('''1. Pandas
2. Selenium
3. Streamlit
4. R и tidyverse
5. ggplot2 с расширениями
6. Машинное обучение: регрессии
7. Графы на networkx
''')


st.header('Регрессии: чем больше мата и громче трек, тем популярней?')
st.markdown('''Пора посмотреть на датасет песен. К сожалению, файл слишком большой, чтобы влезть на GitHub, поэтому для регрессии нам придется оставить только его пятую часть, рандомные 100000 строк. Это сделано с помощью следующего кода:''')

st.code('''
    df_full = pd.read_csv("tracks.csv")
    df_lite = df_full.sample(frac=1)[0:100000]
    df_lite.to_csv('track_lite.csv')
    ''')

st.markdown('''Посмотрим на датасет. Помимо названия, списка исполнителей, даты релиза и популярности тут есть много других метрик. Значение большинства понятно из названия, уточним лишь некоторые: 
mode (0 - мажор, 1 - минор)
key (тональность, от 0 до 11, начиная от C)
explicit (0 - нет контента 18+, 1 - есть)
''')
with st.echo(code_location="above"):
    df = pd.read_csv("tracks_lite.csv")
    df.sort_values(by='popularity', ascending=False)[0:20]

st.markdown('''Теперь можно запускать обучение, чтобы понять, как зависит популярность трека от разных факторов''')
with st.echo(code_location="above"):
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
    regr = LinearRegression()
    df_coefs = pd.DataFrame()
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

st.markdown('''Пока мы строили графики, мы записывали коэффициенты регрессии в датасет. Теперь можно сравнить разные факторы между собой и понять, что больше всего связано с популярностью трека и в какую сторону''')

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    coefs = df_coefs.transpose().sort_values(0).rename(columns={0: "coefficient"})
    plt.plot(1)
    coefs.plot.bar(color = '#33CCCC')
    fig = plt.plot()
    st.pyplot(fig)


st.header('Дальше пока можно не смотреть')

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