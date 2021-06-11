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
from PIL import Image


st.title('Как написать трек, который разорвет чарты: изучаем на данных')
st.header('О проекте и используемых данных')
st.subheader('Датасет')
st.markdown('''Мы будем использовать датасет от Spotify. В нем для всех песен на Spotify указана популярность на апрель 2021 года и множество факторов от 0 до 1, которые автоматические рассчитывает Spotify: в том числе energy, danceability, acousticness и т.д. Скоро мы посмотрим на него поближе, но при желании подробнее про датасет можно прочитать тут: https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=artists.csv''')
st.subheader('Описание проекта')
st.markdown('''Проект состоит из двух частей - R и Python. В первой части с помощью R мы поближе посмотрим на факторы популярности треков и сделаем несколько интересных визуализаций. Во второй с помощью Python построим регрессии для этих факторов и поймем, каким должен быть трек, чтобы с большей вероятностью стать популярным. Также мы построим граф фитов 50 самых популярных исполнителей на Spotify. А в конце спарсим таблицу самых популярных аккаунтов в Instagram из Википедии, чтобы понять, насколько музыканты популярны относительно других знаменитостей''')
st.subheader('Используемые технологии')
st.markdown('''1. Pandas
2. Selenium
3. Streamlit
4. R и tidyverse
5. ggplot2 с расширениями
6. Машинное обучение: регрессии
7. Графы на networkx
''')


st.header(''
          'Регрессии: чем больше мата и громче трек, тем популярней?')
st.markdown('''Пора посмотреть на датасет песен. К сожалению, файл слишком большой, чтобы влезть на GitHub, поэтому для регрессии нам придется оставить только его пятую часть, рандомные 100000 строк. Это сделано с помощью следующего кода:''')

st.code('''
    df_full = pd.read_csv("tracks.csv")
    df_lite = df_full.sample(frac=1)[0:100000]
    df_lite.to_csv('track_lite.csv')
    ''')

st.markdown('''
Посмотрим на датасет. Помимо названия, списка исполнителей, даты релиза и популярности тут есть много других метрик. Значение большинства понятно из названия, уточним лишь некоторые: 
mode (0 - мажор, 1 - минор)
key (тональность, от 0 до 11, начиная от C)
explicit (0 - нет контента 18+, 1 - есть)
''')
with st.echo(code_location="above"):
    df = pd.read_csv("tracks_lite.csv")
    df.sort_values(by='popularity', ascending=False)[0:20]

st.markdown('''
Теперь можно запускать обучение, чтобы понять, как зависит популярность трека от разных факторов''')

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
    df_coefs = pd.DataFrame()
    regr = LinearRegression()
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

st.markdown('''
Пока мы строили графики, мы записывали коэффициенты регрессии в датасет. Теперь можно сравнить разные факторы между собой и понять, что больше всего связано с популярностью трека и в какую сторону''')

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    coefs = df_coefs.transpose().sort_values(0).rename(columns={0: "coefficient"})
    plt.plot(1)
    coefs.plot.bar(color = '#33CCCC')
    fig = plt.plot()
    st.pyplot(fig)


st.header(''
          'Nicki Minaj на фитах: граф совместных треков самых популярных исполнителей')

st.markdown('''Получим данные для графа. Из artists.csv мы возьмем исполнителей с наибольшим числом фолловеров. Из tracks.csv возьмем верхнюю треть самых популярных треков и вытащим из них списки исполнителей. Все это нужно из-за того, что на GitHub целиком файлы не помещаются. 
''')
st.code('''    df_full = pd.read_csv("tracks.csv")
    df_feats = df_full.sort_values(by = 'popularity', ascending = False)['artists']
    df_feats.to_csv('df_feats.csv')

    artists_full = pd.read_csv("artists.csv")
    artists_lite = artists_full.sort_values(by = 'followers', ascending = False)[0:1000]
    artists_lite.to_csv('artists_lite.csv')
''')


with st.echo(code_location="above"):

    artists = pd.read_csv("artists_lite.csv")
    df_feats = pd.read_csv("df_feats.csv")

st.markdown('''
Возьмем 50 артистов с наибольшим количеством фолловеров на Spotify. Это будут вершины графа
    ''')

with st.echo(code_location="above"):
    most_followed_artists = artists.sort_values(by='followers', ascending=False)[0:50]

    vertices = list(most_followed_artists['name'])
    vertices

st.markdown('''
Теперь сделаем список всех фитов с этими звездами, а затем оставим только те, в которых есть два музыканта из топ-50
    ''')
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


st.markdown('''
Немного почистим данные (они в виде строки, а не списка) и уберем из фитов артистов, которые не входят в топ-50''')

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

st.markdown('''
Посмотреть на получившие кортежи фитов можно, раскрыв вывод. Эти кортежи и будут ребрами графа''')

with st.echo(code_location="above"):

    feats_list_tuples = [tuple(el) for el in feats_list]
    feats_list_tuples

st.markdown('''
Следующий код делает нам граф в Jupiter Notebook. Его очень приятно зумить и двигать, но из-за проблем Heroku с networkx сюда можно вставить только скрин результата''')

st.code('''
G = nx.Graph()
G.add_nodes_from(vertices)
G.add_edges_from(feats_list_tuples)
net = Network(width='1000px', notebook=True)
net.from_nx(G)
net.show("feats.html")
''')

st.markdown('''
Есть исполнители вроде Twenty One Pilots, которые предпочитают исполнять свои песни сами. Но есть и группа популярных исполнителей, которые между собой успели записать много совместных треков
''')
img = Image.open('graph.png')
st.image(img, caption='Граф подальше')
img2 = Image.open('graph_closer.png')
st.markdown('''
Если приблизить, становятся видны 2 группы исполнителей: американских и латиноамериканских (J Balvin, Bad Bunny, Daddy Yankee, Ozuna), которые фитуют больше друг с другом. Ну и музыканты вроде Justin Bieber и Nicki Minaj, которые успели записать очень много фитов
''')
st.image(img2, caption='Граф поближе')

st.header(''
          'Музыка - легкий путь к славе? Посмотрим, насколько популярные инстаграмы музыкантов')

st.markdown('С помощью Selenium соберем данные с таблички Instagram-аккаунтов с наибольшим количеством подписчиков')

st.code('''    driver = Chrome(executable_path="chromedriver.exe")

    driver.get("https://en.wikipedia.org/wiki/List_of_most-followed_Instagram_accounts")
    driver.implicitly_wait(2)

    table = driver.find_elements_by_tag_name("tr")
    our_table = table[1:51]

    df_celebs = pd.DataFrame(columns=['instagram', 'name', 'mln_followers', 'field', 'country'])

    for i in range(50):
        string = our_table[i].find_elements_by_tag_name("td")
        insta = string[0].text
        name = string[1].text
        mln_followers = int(float(string[2].text))
        field = string[3].text
        country = string[4].text
        country = country[1:len(country)]
        df_temp = pd.DataFrame(data=[[insta, name, mln_followers, field, country]],
                               columns=['instagram', 'name', 'mln_followers', 'field', 'country'])
        df_celebs = df_celebs.append(df_temp)
        df_celebs = df_celebs.reset_index().drop(columns=['index'])
        ''')


st.markdown(''
            'Получили таблицу таких аккаунтов. Поскольку heroku плохо работает с Selenium, результат выполнения кода мы сохранили локально. Посмотрим на получившийся датафрейм:')

with st.echo(code_location="above"):
    # df_celebs.to_csv('celebs.csv')

    df_celebs = pd.read_csv('celebs.csv')
    df_celebs

st.markdown(''
            'Посмотрим, из каких стран самые популярные знаменитости в инстаграме')

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.plot(1)
    df_celebs['country'].value_counts().sort_values().plot.pie(y='mln_followers', labeldistance=1.1, legend=None,
                                                               figsize=(10, 10))
    fig = plt.plot()
    st.pyplot(fig)

st.markdown(''
            'А вот и музыканты: по сумме подписчиков музыканты в отрыве от остальных родов занятий. При этом музыканты появляются и в других категориях: музыка является трамплином для того, чтобы оказаться в новых сферах')

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df_pie = df_celebs.groupby(['field']).sum(['mln_followers']).sort_values('mln_followers')
    plt.plot(1)
    df_pie.plot.pie(y='mln_followers', labeldistance=1.1, legend=None, figsize=(10, 10))
    fig = plt.plot()
    st.pyplot(fig)

