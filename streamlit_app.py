import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Модель классификации вкладчиков банка", layout="wide")
st.title("Классификация потенциальных вкладчиков")
st.write("Работа с данными клиентов португальского банка, собранными в ходе маркетинговой кампании")

data = pd.read_csv("https://raw.githubusercontent.com/miku-taj/bank_project/refs/heads/master/bank-additional-full.csv", sep=';')

st.sidebar.markdown('''
# Разделы
- [Размер датасета](#razmer-dataseta)
- [Случайные 10 строк](#sluchaynye-10-strok)
''', unsafe_allow_html=True)

# - [Метрики модели](#metriki-modeli)
# - [Сделать прогноз](#sdelat-prognoz)

st.header("Размер датасета")
st.write(f"Строки: {data.shape[0]} Столбцы: {data.shape[1]}")

st.header("Случайные 10 строк")
st.dataframe(data.sample(10), use_container_width=True)

st.header("Описание набора данных")
st.markdown("""
### Входные переменные

##### Данные о клиентах банка
1. **age** — возраст (числовой)  
2. **job** — тип работы (категориальный): `"admin."`, `"blue-collar"`, `"entrepreneur"`, `"housemaid"`, `"management"`, `"retired"`, `"self-employed"`, `"services"`, `"student"`, `"technician"`, `"unemployed"`, `"unknown"`  
3. **marital** — семейное положение (категориальный): `"divorced"`, `"married"`, `"single"`, `"unknown"`  
   *Примечание:* `"divorced"` означает разведён или вдовец/вдова  
4. **education** — образование (категориальный): `"basic.4y"`, `"basic.6y"`, `"basic.9y"`, `"high.school"`, `"illiterate"`, `"professional.course"`, `"university.degree"`, `"unknown"`  
5. **default** — есть ли задолженность по кредиту? (категориальный): `"no"`, `"yes"`, `"unknown"`  
6. **housing** — есть ли ипотечный кредит? (категориальный): `"no"`, `"yes"`, `"unknown"`  
7. **loan** — есть ли личный кредит? (категориальный): `"no"`, `"yes"`, `"unknown"`  

#### Данные последнего контакта текущей кампании
8. **contact** — способ связи (категориальный): `"cellular"`, `"telephone"`  
9. **month** — месяц последнего контакта (категориальный): `"jan"`, `"feb"`, `"mar"`, …, `"nov"`, `"dec"`  
10. **day_of_week** — день недели последнего контакта (категориальный): `"mon"`, `"tue"`, `"wed"`, `"thu"`, `"fri"`  
11. **duration** — длительность последнего контакта в секундах (числовой)  

#### Прочие атрибуты
12. **campaign** — количество контактов в рамках этой кампании с клиентом (числовой, включает последний контакт)  
13. **pdays** — количество дней после последнего контакта с клиентом в предыдущей кампании (числовой; `999` означает, что клиент ранее не контактировался)  
14. **previous** — количество контактов до этой кампании (числовой)  
15. **poutcome** — результат предыдущей кампании (категориальный): `"failure"`, `"nonexistent"`, `"success"`  

#### Социально-экономические показатели
16. **emp.var.rate** — коэффициент изменения занятости, квартальный показатель (числовой)  
17. **cons.price.idx** — индекс потребительских цен, ежемесячный показатель (числовой)  
18. **cons.conf.idx** — индекс потребительской уверенности, ежемесячный показатель (числовой)  
19. **euribor3m** — ставка Euribor 3 месяца, ежедневный показатель (числовой)  
20. **nr.employed** — количество сотрудников, квартальный показатель (числовой)  

#### Выходная переменная (целевой признак)
21. **y** — оформил ли клиент депозит? (бинарный): `"yes"`, `"no"`
""")

st.write('В рамках настоящего проекта стоит цель определить, оформит ли депозит клиент без информации о предыдущей истории клиента. Мы удалим признаки, имеющий отношение к предыдущей кампании, такие как pdays, previous, poutcome.')

st.header("Визуализация")
palette = sns.color_palette("Set2", n_colors=8).as_hex()

st.subheader('Распределение целевой переменной')
sns.set_theme(style="whitegrid", palette="Set2", font_scale=0.9)

fig = px.histogram(data, x='y', color='y', barmode='stack', color_discrete_sequence=palette,
                   text_auto=True, width=500, height=400)
st.plotly_chart(fig, use_container_width=False)

st.write('Классы распределены неравномерно.')

st.subheader('Демография клиентов, оформивших и не оформивших займ')

# Figure 1: job 
fig1 = px.histogram(
    data, 
    y="job", 
    color="y", 
    barmode="stack",
    color_discrete_sequence=palette,
    text_auto=True,
    height=400,
)
fig1.update_layout(
    title="Сравнение числа клиентов и вкладчиков в разрезе профессий",
    xaxis_title="",
    yaxis_title="",
    showlegend=True
)

# --- Figure 2: marital ---
fig2 = px.histogram(
    data,
    y="marital",
    color="y",
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig2.update_layout(
    title="Сравнение числа клиентов и вкладчиков в разрезе <br>семейного положения",
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)

# --- Figure 3: education ---
fig3 = px.histogram(
    data,
    y="education",
    color="y",
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig3.update_layout(
    title="Сравнение числа клиентов и вкладчиков в разрезе <br>уровня образования",
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)
st.plotly_chart(fig1, use_container_width=True)
# --- Display in Streamlit using columns ---
col1, col2 = st.columns([1, 1])  # first plot wider
with col1:
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    st.plotly_chart(fig3, use_container_width=True)


st.subheader('Другие характеристики клиентов, оформивших и не оформивших займ')

fig1 = px.histogram(
    data, 
    y="loan", 
    color="y", 
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig1.update_layout(
    title="Клиенты и вкладчики по наличию \nперсонального займа",
    xaxis_title="",
    yaxis_title="",
    showlegend=True
)

# --- Figure 2: marital ---
fig2 = px.histogram(
    data,
    y="default",
    color="y",
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig2.update_layout(
    title="Клиенты и вкладчики по факту \nкредитного дефолта",
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)

# --- Figure 3: education ---
fig3 = px.histogram(
    data,
    y="housing",
    color="y",
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig3.update_layout(
    title="Клиенты и вкладчики по наличию \nжилищного кредита",
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)

col1, col2, col3 = st.columns([1, 1, 1]) 
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)


st.subheader('Коммуникация с клиентами, оформившими и не оформившими займ')


fig1 = px.histogram(
    data, 
    y="contact", 
    color="y", 
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig1.update_layout(
    title="Клиенты и вкладчики по типу <br>связи",
    xaxis_title="",
    yaxis_title="",
    showlegend=True
)

# --- Figure 2: marital ---
fig2 = px.histogram(
    data,
    y="month",
    color="y",
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig2.update_layout(
    title="Клиенты и вкладчики по месяцу <br>последнего контакта",
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)

# --- Figure 3: education ---
fig3 = px.histogram(
    data,
    y="day_of_week",
    color="y",
    color_discrete_sequence=palette,
    barmode="stack",
    text_auto=True,
    height=300,
)
fig3.update_layout(
    title="Клиенты и вкладчики по \nдню недели последнего контакта",
    xaxis_title="",
    yaxis_title="",
    showlegend=False
)

col1, col2, col3 = st.columns([1, 1, 1]) 
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

st.subheader('')


