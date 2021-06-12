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
          'Часть первая. R')
st.markdown('''
Поскольку R не деплоится очевидным образом на heroku, для этой части я буду приводить только код, а результат его исполнения, сохраненный заранее, буду подгружать отдельно. ''')

st.markdown('''Весь датасет слишком большой для GitHub, поэтому для первого знакомства мы посмотрим только на первые 100 строк, отсортировав по популярности исходный файл. Весь анализ на R сделан на основе полного исходного файла tracks.csv. Следующий код сделает нам датасет для ознакомления:
''')
st.code('''
df_full = pd.read_csv("tracks.csv")
df_small = df_full.sort_values(by = 'popularity', ascending = False)[0:100]
df_small.to_csv('tracks_first_100.csv')
''')
st.markdown('''
А вот таким кодом мы начнем проект и выведем датасет в R:
''')
st.code('''
library(tidyverse)
library(ggridges)
library(ggthemes)

dat = read_csv("tracks.csv")
dat = dat %>% drop_na()

dat %>% arrange(desc(popularity)) %>% head(100)
''')
df_first_100 = pd.read_csv("tracks_first_100.csv")
df_first_100

st.markdown('''
Посмотрим, как меняются характеристики треков, если в них есть контент 18+ (explicit = 1) или нет (explicit = 0)
''')
st.code('''
explicit_or_not = dat %>% group_by(explicit) %>% summarise(mean(popularity), mean(duration_ms), mean(danceability), mean(energy), mean(mode), mean(acousticness), mean(instrumentalness), mean(tempo))
write.csv(explicit_or_not, 'explicit_or_not.csv')
''')
explicit = pd.read_csv('explicit_or_not.csv')
explicit

st.markdown('''
Видим, что треки с контентом 18+ в среднем более энергичны. Посмотрим на распределение энергичности в зависимости от explicit, разбив дополнительно на мажорные (сверху) и минорные (снизу)
''')
st.code('''
dat %>% ggplot(aes(x = energy, y = mode, group = mode, fill = explicit)) + 
  geom_density_ridges(size = 1, color = 'black', alpha = 0.8) +
  theme_bw() +
  facet_wrap(~explicit) +
  labs(title =  'Песни с запрещенным контентом более энергичны', subtitle = 'При этом минорные треки даже немного энергичней мажорных')
''')
img1 = Image.open('img1.png')
st.image(img1)

st.markdown('''
Теперь возьмем 5000 самых популярных треков. Посмотрим на их распределение в зависимости от танцевальности, минора/мажора и наличия explicit контента
''')
st.code('''
dat %>% arrange(desc(popularity)) %>% head(5000) %>%
  ggplot(aes(x = danceability,y = popularity, alpha = 1, color = explicit)) +
  geom_point() +
  theme_bw() +
  geom_smooth(colour = 'orange', size = 1.5) +
  facet_wrap(~mode) +
  scale_x_continuous(limits = c(0.25, 1), expand = c(0, 0)) + 
  labs(title = '5000 самых популярных треков: минор и мажор', subtitle =  'Мажорных треков больше, их популярность меньше зависит от танцевальности 
Видно, что треки 18+ смещены в сторону большей танцевальности')
''')
img2 = Image.open('img2.png')
st.image(img2)

st.header(''
          'Часть вторая. Python')

st.header(''
          'Регрессии: чем больше мата и громче трек, тем популярней?')
st.markdown('''Так как исходный файл слишком большой, для регрессии нам придется оставить только его пятую часть, рандомные 100000 строк. Это сделано с помощью следующего кода:''')

st.code('''
    df_full = pd.read_csv("tracks.csv")
    df_lite = df_full.sample(frac=1)[0:100000]
    df_lite.to_csv('track_lite.csv')
    ''')

st.markdown('''
Посмотрим еще раз на датасет
''')
with st.echo(code_location="above"):
    df = pd.read_csv("tracks_lite.csv")
    df.sort_values(by='popularity', ascending=False, ignore_index=True)[0:20]

st.markdown('''
Теперь можно запускать обучение, чтобы понять, как зависит популярность трека от разных факторов''')

with st.echo(code_location="above"):
    columns = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']
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

st.markdown('''
Итак, лучше забыть про акустику и инструментальность, если хочешь написать популярный трек. Не пиши мелодию в миноре, но темп, тональность и длительность можешь выбрать любую. Не забудь добавить в песню такого текста, чтобы детям слушать было нельзя, - это ключ к успеху
''')
st.header('')
st.header('Nicki Minaj на фитах: граф совместных треков самых популярных исполнителей')

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
            'Посмотрим, у какого рода занятий больше всего подписчиков')
st.markdown(''
            'По сумме подписчиков музыканты в отрыве от остальных родов занятий. При этом музыканты появляются и в других категориях: музыка является трамплином для того, чтобы оказаться в новых сферах')

with st.echo(code_location="above"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df_pie = df_celebs.groupby(['field']).sum(['mln_followers']).sort_values('mln_followers')
    plt.plot(1)
    df_pie.plot.pie(y='mln_followers', labeldistance=1.1, legend=None, figsize=(10, 10))
    fig = plt.plot()
    st.pyplot(fig)

