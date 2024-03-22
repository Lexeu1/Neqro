import numpy as np # импорт библиотеки numpy и присвоение ей псевдонима np
import pandas as pd # импорт библиотеки pandas и присвоение ей псевдонима pd
pd.set_option("display.max.columns", 100) # установка опции для pandas отображать максимум 100 столбцов
%matplotlib встроенные предупреждения импорта # включение встроенных предупреждений matplotlib
import matplotlib.pyplot as plt # импорт библиотеки matplotlib и присвоение ей псевдонима plt
import seaborn as sns # импорт библиотеки seaborn и присвоение ей псевдонима sns
alerts.filterwarnings("игнорировать")  # игнорировать предупреждения

DATA_URL = " https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/ " # ссылка на данные
data = pd.read_csv(DATA_URL + "adult.data.csv") # чтение данных из csv файла
data.head() # отображение первых строк данных

# Ответы на вопросы:

# Сколько мужчин и женщин представлено в этом наборе данных?
gender_counts = data['sex'].value_counts()
print(gender_counts)

# Каковы женщины среднего возраста (возрастная характеристика)?
average_female_age = data[data['sex'] == 'Female']['age'].mean()
print(average_female_age)

# Каков процент граждан Германии?
german_citizens_percentage = (data['native-country'] == 'Germany').sum() / data.shape[0] * 100
print(german_citizens_percentage)

# Среднее значение и стандартное отклонение для каждой категории заработка
mean_std_above_50k = data[data['income'] == '>50K']['age'].agg(['mean', 'std'])
mean_std_below_50k = data[data['income'] == '<=50K']['age'].agg(['mean', 'std'])
print("Mean and std deviation for income above 50k:", mean_std_above_50k)
print("Mean and std deviation for income below 50k:", mean_std_below_50k)

# Правда ли, что люди с доходом более 50k имеют как минимум бакалавриат?
high_income_education = data[data['income'] == '>50K']['education'].unique()
print("High income education levels:", high_income_education)

# Статистика по возрасту для каждой расы и пола
sns.catplot(x='race', y='age', hue='sex', kind='box', data=data)

# Максимальный возраст мужчин американо-индейско-эскимосской расы
max_age_american_indian = data[(data['race'] == 'Amer-Indian-Eskimo') & (data['sex'] == 'Male')]['age'].max()
print("Max age of male American-Indian-Eskimo race:", max_age_american_indian)

# Максимальный возраст для женщин из Азиатско-Тихоокеанского региона
max_age_asian_pacific = data[(data['race'] == 'Asian-Pac-Islander') & (data['sex'] == 'Female')]['age'].max()
print("Max age of female Asian-Pac-Islander race:", max_age_asian_pacific)

# Прогнозирование, превышает ли доход $50k
# Для прогнозирования можно использовать различные модели машинного обучения, например, логистическую регрессию или случайный лес. Можно провести обучение на данных и оценить точность прогнозов.
