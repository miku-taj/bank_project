import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import shap
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
- [Описание набора данных](##opisanie-nabora-dannykh)
- [Визуализация](#vizualizatsiya)
- [Предобработка](#predobrabotka)
- [Метрики бейзлайн моделей](#metriki-beyzlayn-modeley)
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

st.subheader('Демография клиентов, оформивших и не оформивших депозит')
# Figure: age 
fig1 = px.histogram(
    data, 
    x="age", 
    color="y", 
    barmode="stack",
    color_discrete_sequence=palette,
    text_auto=True,
    height=400,
)
fig1.update_layout(
    title="Сравнение числа клиентов и вкладчиков в разрезе возраста",
    xaxis_title="",
    yaxis_title="",
    showlegend=True
)

st.plotly_chart(fig1, use_container_width=True)
st.write(f"Нет видимой связи между возрастом и фактом оформления кредита. Кроме того, корреляция близка к 0 (", round(data['age'].corr(data['y'].apply(lambda x: 0 if x == 'no' else 1)), 3), ")")

# Figure: job 
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

# Figure: marital
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

# Figure: education
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


st.subheader('Другие характеристики клиентов, оформивших и не оформивших депозит')

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
    title="Клиенты и вкладчики по наличию <br>персонального займа",
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
    title="Клиенты и вкладчики по факту <br>кредитного дефолта",
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
    title="Клиенты и вкладчики по наличию <br>жилищного кредита",
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
st.write("Из графиков следует, что наличие персонального или жилищного займа мало/не влияет на оформление депозита. Количество клиентов в кредитном дефолте очень мало, оформленных депозитов нет. ")

st.subheader('Коммуникация с клиентами, оформившими и не оформившими депозит')


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
    title="Клиенты и вкладчики по <br>дню недели последнего контакта",
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

def encode_education(x):
    digit = re.search(r'\d', x)
    if digit:
        return int(digit.group())
    if x == 'illiterate':
        return 0
    if x == 'high.school':
        return 12
    if x == 'professional.course':
        return 14
    if x == 'university.degree':
        return 16 
enc_data = data.copy()
enc_data['education'] = enc_data['education'].apply(encode_education)
enc_data['y'] = enc_data['y'].apply(lambda x: 0 if x == 'no' else 1)

corr_matrix = data.select_dtypes(include=['int', 'float']).corr()

fig = px.imshow(
    corr_matrix,
    text_auto=True,          
    color_continuous_scale='RdBu_r',
    aspect='auto',
    height=400
)

fig.update_layout(
    title="График корреляции",
    xaxis_title="Признаки",
    yaxis_title="Признаки"
)
st.plotly_chart(fig, use_container_width=False)


st.header('Предобработка')
st.write('Удалены пропущенные значения. Признак education переведен в числовой формат (кол-во лет обучения). Остальные категориальные признаки закодированы с помощью One Hot Encoding (что обусловлено небольшим кол-вом уникальных значений в каждой категории) и нормализованы для обучения базовых моделей - Логистической Регрессии, К Ближайших Соседей, Дерева Решений, а также модели Случайного Леса.')
st.write('Данные разделены на тренировочный, валидационный и тестовый датасеты в примерном соотношении 60%:20%:20%')

data = data.drop(['pdays', 'poutcome', 'previous'], axis=1)

data = data.replace('unknown', np.nan)
data = data.dropna(how='any')

X = data.drop(['y', 'duration'], axis=1)
y = data['y']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.25,
                                                    random_state=42,
                                                    stratify=y_train)

cat_cols = X.select_dtypes(include=object).columns
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train1_cat = ohe.fit_transform(X_train1[cat_cols])
X_val_cat  = ohe.transform(X_val[cat_cols])
X_test_cat  = ohe.transform(X_test[cat_cols])

feature_names = ohe.get_feature_names_out(cat_cols)

X_train1_cat_df = pd.DataFrame(X_train1_cat, columns=feature_names, index=X_train1.index)
X_val_cat_df  = pd.DataFrame(X_val_cat, columns=feature_names, index=X_val.index)
X_test_cat_df  = pd.DataFrame(X_test_cat, columns=feature_names, index=X_test.index)


X_train1_ohe = pd.concat((X_train1.drop(cat_cols, axis=1), X_train1_cat_df), axis=1)
X_val_ohe = pd.concat((X_val.drop(cat_cols, axis=1), X_val_cat_df), axis=1)
X_test_ohe = pd.concat((X_test.drop(cat_cols, axis=1), X_test_cat_df), axis=1)

ss = StandardScaler()

X_train1_scaled = pd.DataFrame(ss.fit_transform(X_train1_ohe), columns=X_train1_ohe.columns)
X_val_scaled =  pd.DataFrame(ss.transform(X_val_ohe), columns=X_val_ohe.columns)
X_test_scaled =  pd.DataFrame(ss.transform(X_test_ohe), columns=X_test_ohe.columns)

models = {
    'LogisticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier()
}


res = {}

for name, model in models.items():

    res[name] = {}

    model.fit(X_train1_scaled, y_train1)
    train1_proba = model.predict_proba(X_train1_scaled)
    val_proba = model.predict_proba(X_val_scaled)
    test_proba = model.predict_proba(X_test_scaled)

    train1_roc_auc = roc_auc_score(y_train1, train1_proba[:, 1])
    val_roc_auc = roc_auc_score(y_val, val_proba[:, 1])
    test_roc_auc = roc_auc_score(y_test, test_proba[:, 1])

    res[name]['Train ROC AUC'] = train1_roc_auc
    res[name]['Val ROC AUC'] = val_roc_auc
    res[name]['Train-Val Difference'] = train1_roc_auc - val_roc_auc
    res[name]['Test ROC AUC'] = test_roc_auc

MODELS_METRICS = pd.DataFrame(res).T

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train1_scaled, y_train1)

res = {
    'RandomForrest': {}
}

train1_proba = rf.predict_proba(X_train1_scaled)
val_proba = rf.predict_proba(X_val_scaled)
test_proba = rf.predict_proba(X_test_scaled)

train1_roc_auc = roc_auc_score(y_train1, train1_proba[:, 1])
val_roc_auc = roc_auc_score(y_val, val_proba[:, 1])
test_roc_auc = roc_auc_score(y_test, test_proba[:, 1])

res['RandomForrest']['Train ROC AUC'] = train1_roc_auc
res['RandomForrest']['Val ROC AUC'] = val_roc_auc
res['RandomForrest']['Train-Val Difference'] = train1_roc_auc - val_roc_auc
res['RandomForrest']['Test ROC AUC'] = test_roc_auc

MODELS_METRICS = pd.concat((MODELS_METRICS, pd.DataFrame(res).T), axis=0)

st.header('Метрики бейзлайн моделей')
st.dataframe(MODELS_METRICS, use_container_width=True)

cat_cols = X.select_dtypes(include=object).columns
cat_cols_id = [X.columns.get_loc(col) for col in cat_cols]

catboost_model = CatBoostClassifier(
    iterations=856,
    depth=6,
    learning_rate=0.05, 
    l2_leaf_reg=5, 
    eval_metric='AUC',
    verbose=100, 
    random_state=42, 
)

catboost_model.fit(
    X_train1, y_train1, 
    cat_features=cat_cols_id,
    eval_set=(X_val, y_val)
)

train1_proba = catboost_model.predict_proba(X_train1)
val_proba = catboost_model.predict_proba(X_val)
test_proba = catboost_model.predict_proba(X_test)

train1_roc_auc = roc_auc_score(y_train1, train1_proba[:, 1])
val_roc_auc = roc_auc_score(y_val, val_proba[:, 1])
test_roc_auc = roc_auc_score(y_test, test_proba[:, 1])

res = {
  'Catboost Full': {}
}
res['Catboost Full']['Train ROC AUC'] = train1_roc_auc
res['Catboost Full']['Val ROC AUC'] = val_roc_auc
res['Catboost Full']['Train-Val Difference'] = train1_roc_auc - val_roc_auc
res['Catboost Full']['Test ROC AUC'] = test_roc_auc

MODELS_METRICS = pd.concat((MODELS_METRICS, pd.DataFrame(res).T), axis=0)


st.header('Отбор признаков')
st.subheader('Важность признаков')
importances = catboost_model.get_feature_importance(type="FeatureImportance")
feature_names = list(X_train.columns)

idx = np.argsort(importances)[::-1]
sorted_importances = np.array(importances)[idx]
sorted_features = [feature_names[i] for i in idx]

fig = px.bar(
    y=sorted_features,
    x=sorted_importances,
    labels={'y': 'Признак', 'x': 'Важность'},
    title="CatBoost Feature Importances"
)


explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer(X_train, y_train)

plt.figure(figsize=(10,6))
shap.summary_plot(
    shap_values.values,         
    shap_values.data,           
    feature_names=shap_values.feature_names,
    plot_type="dot",            
    max_display=20,
    show=False                  
)
plt.title('SHAP значения признаков')


col1, col2 = st.columns([1, 1]) 
with col1:
    st.plotly_chart(fig)
with col2:
    st.pyplot(plt.gcf())
st.write("Исходя из важности признаков и принимая во внимание SHAP значения, были удалены признаки default, loan, housing, education.")

X_train1, X_val, X_test = X_train1.drop(['default','loan','housing','education'], axis=1), X_val.drop(['default','loan','housing','education'], axis=1), X_test.drop(['default','loan','housing','education'], axis=1)

cat_cols = X_test.select_dtypes(include=object).columns
cat_cols_id = [X_test.columns.get_loc(col) for col in cat_cols]

catboost_model.fit(
    X_train1, y_train1, 
    cat_features=cat_cols_id,
    eval_set=(X_val, y_val)
)

train1_proba = catboost_model.predict_proba(X_train1)
val_proba = catboost_model.predict_proba(X_val)
test_proba = catboost_model.predict_proba(X_test)

train1_roc_auc = roc_auc_score(y_train1, train1_proba[:, 1])
val_roc_auc = roc_auc_score(y_val, val_proba[:, 1])
test_roc_auc = roc_auc_score(y_test, test_proba[:, 1])

res['Catboost'] = {}
res['Catboost']['Train ROC AUC'] = train1_roc_auc
res['Catboost']['Val ROC AUC'] = val_roc_auc
res['Catboost']['Train-Val Difference'] = train1_roc_auc - val_roc_auc
res['Catboost']['Test ROC AUC'] = test_roc_auc
MODELS_METRICS = pd.concat((MODELS_METRICS, pd.DataFrame(res).T), axis=0)
st.dataframe(MODELS_METRICS)

explainer = shap.TreeExplainer(catboost_model)

st.header('Сделать прогноз')

with st.form("user_input_form"):

    age_input = st.number_input("Возраст (Age)", min_value=int(data['age'].min()), max_value=int(data['age'].max()), value=int(data['age'].median()), step=1)   
    marital_input = st.radio("Семейное положение (Marital)", list(data['marital'].value_counts().sort_values(ascending=False).index))
    job_input = st.selectbox("Профессия (Job)", list(data['job'].value_counts().sort_values(ascending=False).index), index=0)
  
    contact_input = st.radio("Тип связи (Contact)", list(data['contact'].value_counts().sort_values(ascending=False).index))
    month_input = st.selectbox("Месяц последнего контакта (Month)", list(data['month'].value_counts().sort_values(ascending=False).index), index=0)
    dow_input = st.selectbox("День недели последнего контакта (Day Of Week)", list(data['day_of_week'].value_counts().sort_values(ascending=False).index), index=0)

    col1, col2, col3 = st.columns([1, 1, 1]) 
    with col1:
      campaign_input = st.number_input('Количество контактов в рамках этой кампании с клиентом (campaign)', value=int(data['campaign'].median()))
      euribor3m_input = st.number_input('Ставка Euribor 3 месяца, ежедневный показатель (euribor3m)', value=float(data['euribor3m'].median()))
    with col2:
      emp_var_input = st.number_input('Коэффициент изменения занятости, квартальный показатель (emp.var.rate)', value=float(data['emp.var.rate'].median()))
      employed_input = st.number_input('Количество сотрудников, квартальный показатель (nr.employed)', value=float(data['nr.employed'].median()))
    with col3:
      cons_price_input = st.number_input('Индекс потребительских цен, ежемесячный показатель (cons.price.idx)', value=float(data['cons.price.idx'].median()))
      cons_conf_input = st.number_input('Индекс потребительской уверенности, ежемесячный показатель (cons.conf.idx)', value=float(data['cons.conf.idx'].median()))
    submit_button = st.form_submit_button("Предсказать")

    if submit_button:
        user_input = pd.DataFrame([{
            'age': age_input,
            'job': job_input,
            'marital': marital_input,
            'contact': contact_input,
            'month': month_input,
            'day_of_week': dow_input,
            'campaign': campaign_input,
            'emp.var.rate': emp_var_input,
            'cons.price.idx': cons_price_input,
            'cons.conf.idx': cons_conf_input,
            'euribor3m': euribor3m_input,
            'nr.employed': employed_input
        }])

        user_input = user_input[X_test.columns]
        with st.expander('Просмотреть результат:'):
            pred = catboost_model.predict(user_input)[0]
            if pred == 1:
                st.write(f"**Вероятно, что клиент оформит депозит в рамках текущей кампании. Вероятность равна {model.predict_proba(user_input_scaled)[0][1]}.**" )
            else:
                st.write(f"**Вероятно, что клиент не оформит депозит в рамках текущей кампании {model.predict_proba(user_input_scaled)[0][0]}.**")
            
            shap_values_row = explainer(user_input).values[0]   
            features_row = user_input.iloc[0]                  

            df_plot = pd.DataFrame({
              "feature": X_test.columns,
              "shap_value": shap_values_row,
              "feature_value": features_row.values
            })

            df_plot = df_plot.reindex(df_plot.shap_value.abs().sort_values(ascending=True).index)
            fig = px.bar(
              df_plot,
              x="shap значение",
              y="признак",
              orientation="h",
              color="feature_value",                 
              color_continuous_scale="RdBu",
              title="SHAP значения для записи",
              height=500
            )



