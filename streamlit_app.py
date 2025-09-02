import streamlit as st

st.set_page_config(page_title="Модель классификации вкладчиков банка", layout="wide")
st.title("Классификация потенциальных вкладчиков")
st.write("Работа с данными клиентов португальского банка, собранными в ходе маркетинговой кампании")

data = pd.read_csv("https://raw.githubusercontent.com/miku-taj/titanic_survival_model/refs/heads/master/Clean-Titanic-Dataset.csv")

st.header("Размер датасета")
st.write(f"Строки: {data.shape[0]} Столбцы: {data.shape[1]}")

st.header("Случайные 10 строк")
st.dataframe(data.sample(10), use_container_width=True)

st.header("Визуализация")

sns.set_theme(style="whitegrid", palette="Set2", font_scale=0.9)

fig = plt.figure(figsize=(8, 6))
plt.title('Распределение клиентов, \nоформивших и не оформивших займ')
sns.countplot(data=data, x='y', hue='y', alpha=1.0, stat="percent")
plt.ylabel('Процент клиентов')
plt.xlabel('Оформили займ')

st.pyplot(fig, use_container_width=True)

st.write('Классы распределены неравномерно.')

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title('Сравнение числа клиентов и вкладчиков в разрезе профессий')
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
sns.histplot(data=data, y='job', hue='y', multiple='stack', ax=ax1, alpha=1.0)

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Сравнение числа клиентов и вкладчиков в разрезе \nсемейного положения')
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
sns.histplot(data=data, y='education', hue='y', multiple='stack', ax=ax2, alpha=1.0)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Сравнение числа клиентов и вкладчиков в разрезе \nуровня образования')
ax3.set_xlabel(' ')
ax3.set_ylabel(' ')
sns.histplot(data=data, y='education', hue='y', multiple='stack', ax=ax3, alpha=1.0)
plt.tight_layout()

st.pyplot(fig, use_container_width=True)

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Клиенты и вкладчики по наличию \nперсонального займа')
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
sns.histplot(data=data, x='loan', hue='y', multiple='stack', ax=ax1, alpha=1.0)

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('Клиенты и вкладчики по факту \nкредитного дефолта')
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
sns.histplot(data=data, x='default', hue='y', multiple='stack', ax=ax2, alpha=1.0)

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('Клиенты и вкладчики по наличию \nжилищного кредита')
ax3.set_xlabel(' ')
ax3.set_ylabel(' ')
sns.histplot(data=data, x='housing', hue='y', multiple='stack', ax=ax3, alpha=1.0)
plt.tight_layout()

st.pyplot(fig, use_container_width=True)

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
