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
- [Визуализация](#vizualizatsiya)
''', unsafe_allow_html=True)

# - [Метрики модели](#metriki-modeli)
# - [Сделать прогноз](#sdelat-prognoz)

st.header("Размер датасета")
st.write(f"Строки: {data.shape[0]} Столбцы: {data.shape[1]}")

st.header("Случайные 10 строк")
st.dataframe(data.sample(10), use_container_width=True)

st.header("Визуализация")

st.subheader('Распределение целевой переменной')
sns.set_theme(style="whitegrid", palette="Set2", font_scale=0.9)

fig = px.histogram(data, x='y', color='y', barmode='stack', 
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
    barmode="stack",
    text_auto=True,
    height=300,
)
fig3.update_layout(
    title="Сравнение числа клиентов и вкладчиков в разрезе ,br.уровня образования",
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
    barmode="stack",
    text_auto=True,
    height=400,
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
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Клиенты и вкладчики по типу связи')
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
sns.histplot(data=data, x='contact', hue='y', multiple='stack', ax=ax1, alpha=1.0)

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('Клиенты и вкладчики по месяцу \nпоследнего контакта')
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
sns.histplot(data=data, x='month', hue='y', multiple='stack', ax=ax2, alpha=1.0)

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('Клиенты и вкладчики по \nдню недели последнего контакта')
ax3.set_xlabel(' ')
ax3.set_ylabel(' ')
sns.histplot(data=data, x='day_of_week', hue='y', multiple='stack', ax=ax3, alpha=1.0)
plt.tight_layout()

st.pyplot(fig, use_container_width=True)
