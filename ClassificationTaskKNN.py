#!/usr/bin/env python
# coding: utf-8

# <h1>Прогнозирование оттока клиентов банка</h1>

# <h2>Описание задачи</h2>
# <p>Собрана база клиентов банка (10000 записей). Необходимо спрогнозировать, кто из клиентов собирается отказаться от услуг банка. Каждый клиент Existing Customer (существующий клиент) или Attrited Customer (ушедший клиент).  Таким образом, поставлена задача классификации. Для этой задачи будем использовать метод k-ближайших соседей.</p>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Загрузка данных</h2>
# <p>Получаем данные из файла формата csv и сразу удаляем не используемые последние 2 столбца.</p>

# In[2]:


url = "https://raw.githubusercontent.com/BakanovMaksim/MachineLearning/main/BankChurners.csv"
buffer = pd.read_csv(url)
data = buffer.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)


# <h2>Просмотр первых строк таблицы</h2>

# In[3]:


data.head()


# <ul>
#     <li><strong>CLIENTNUM</strong> - номер клиента (уникальный идентификатор)</li>
#     <li><strong>Attrition_Flag</strong> - активность клиента <strong>Class</strong> (Existing Customer-счет существует, Attrited Customer-счет закрыт)</li>
#     <li><strong>Customer_Age</strong> - возраст клиента</li>
#     <li><strong>Dependent_count</strong> - демографическая переменная</li>
#     <li><strong>Education_Level</strong> - образовательная квалификация клиента</li>
#     <li><strong>Marital_Status</strong> - семейное положение клиента</li>
#     <li><strong>Income_Category</strong> - категория годового дохода клиента</li>
#     <li><strong>Card_Category</strong> - тип банковской карты</li>
#     <li><strong>Months_on_book</strong> - срок отношений с банком</li>
#     <li><strong>Total_Relationship_Count</strong> - количество активных продуктов банка</li>
#     <li><strong>Months_Inactive_12_mon</strong> - количество месяцев бездействия за последние 12 месяцев</li>
#     <li><strong>Contacts_Count_12_mon</strong> - количество контактов за последние 12 месяцев</li>
#     <li><strong>Credit_Limit</strong> - кредитный лимит по банковской карте</li>
#     <li><strong>Total_Revolving_Bal</strong> - общий возобновляемый остаток на кредитной карте</li>
#     <li><strong>Avg_Open_To_Buy</strong> - количество использования карты для покупки (в среднем за 12 месяцев)</li>
#     <li><strong>Total_Amt_Chng_Q4_Q1</strong> - изменение суммы транзакции</li>
#     <li><strong>Total_Trans_Amt</strong> - общая сумма транзакций (последние 12 месяцев)</li>
#     <li><strong>Total_Trans_Ct</strong> - общее количество транзакций (последние 12 месяцев)</li>
#     <li><strong>Total_Ct_Chng_Q4_Q1</strong> - изменение количества транзакции</li>
#     <li><strong>Avg_Utilization_Ratio</strong> - средний коэффициент использования карты</li>
# </ul>

# <strong>Сводная информация о категориальных признаках</strong>

# In[4]:


data.describe()


# <strong>Сводная информация о количественных признаках</strong>

# In[5]:


data.describe(include=['object'])


# <h2>Обработка пропущенных значений</h2>

# In[6]:


data.isna().sum()


# <p>Пропущенных значений нет, поэтому дальнейших действий не требуется</p>

# <h2>Обработка категориальных признаков</h2>

# <p>Для бинарных категориальных признаков используется метод бинаризации, а для не бинарных категориальных признаков используется метод векторизации.</p>

# In[7]:


data_noncategorial = data.copy()
data_noncategorial["Attrition_Flag"] = pd.factorize(data["Attrition_Flag"])[0]
data_noncategorial["Gender"] = pd.factorize(data["Gender"])[0]

categorical_columns = [c for c in data.columns.drop(["Attrition_Flag", "Gender"]) if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']

for column in categorical_columns:
    data_noncategorial = pd.concat((data_noncategorial, pd.get_dummies(data[column])), axis=1)

data_noncategorial.columns


# <h2>Разбиение данных на тестовые и тренировочные</h2>

# Перед обработкой данных из датасета, разделим на обучающую и тестовую выборки, чтобы произвести обработку данных только для обучающей выборки.

# In[8]:


X = data_noncategorial.drop(["CLIENTNUM", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", "Attrition_Flag"], axis=1)
y = data_noncategorial["Attrition_Flag"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape

print(N_train, N_test)


# <h2>Визуализация данных</h2>

# In[9]:


np.random.seed(42)

random_subset = np.random.choice(np.arange(data.shape[0]), size=1000, replace=False)
plt.scatter(data.iloc[random_subset]["Months_on_book"], data.iloc[random_subset]["Attrition_Flag"], alpha = .3)
plt.xlabel("Срок отношений с банком")
plt.ylabel("Активность клиента")
plt.title("Срок отношений")
pass


# In[10]:


plt.scatter(data.iloc[random_subset]["Total_Trans_Amt"], data.iloc[random_subset]["Attrition_Flag"], alpha = .3)
plt.xlabel("Общая сумма транзакций")
plt.ylabel("Активность клиента")
plt.title("Сумма транзакций")
pass


# In[11]:


plt.scatter(data.iloc[random_subset]["Avg_Utilization_Ratio"], data.iloc[random_subset]["Attrition_Flag"], alpha = .3)
plt.xlabel("Ср. коэф. использования карты")
plt.ylabel("Активность клиента")
plt.title("Использование карты")
pass


# In[12]:


plt.figure(figsize = (10, 8))
sns.scatterplot(x="Total_Trans_Amt", y="Avg_Open_To_Buy", size="Card_Category", hue="Attrition_Flag", data=data.iloc[random_subset], alpha=0.7)
pass


# По диаграммам рассеивания, видно, что у уходящих клиентов показатели ниже, чем у активных

# <h2>Матрица корреляции</h2>

# Вычисление матрицы корреляции для обучающей выборки

# In[13]:


corr_mat = X_train.corr()
corr_mat


# <p>Визуализируем матрицу корреляции</p>

# In[14]:


sns.heatmap(corr_mat, square=True, cmap='coolwarm')
pass


# In[15]:


corr_mat > 0.5


# In[16]:


corr_mat.where(np.triu(corr_mat > 0.5, k=1)).stack().sort_values(ascending=False)


# <h2>Нормализация количественных признаков</h2>
# <p>Алгоритм <strong>метода ближайших соседей</strong> чувствителен к масштабированию данных.</p>
# <p>Поскольку у количественных признаков схожий физический смысл, в нашем случае не нужна нормализация признаков.</p>

# <h2>Обучение модели</h2>

# <strong>Реализация функций обучения модели и вывода результатов</strong>

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def fitKnn(n):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def printKnnResult():
    count = 0
    for value in y_test_predict:
        if value == 1:
            count = count + 1
    print("Общее количество клиентов - ", y_test_predict.size)
    print("Количество уходящих клиентов - ", count)
    
def printError():
    err_test  = np.mean(y_test  != y_test_predict)
    print("Ошибка -", err_test)
    print("Матрица рассогласования:")
    print(confusion_matrix(y_test, y_test_predict))


# <strong>Количество соседей = 2</strong>

# In[23]:


y_test_predict = fitKnn(2)
printKnnResult()
printError()


# <strong>Количество соседей = 9</strong>

# In[24]:


y_test_predict = fitKnn(9)
printKnnResult()
printError()


# <strong>Количество соседей = 16</strong>

# In[25]:


y_test_predict = fitKnn(16)
printKnnResult()
printError()


# <strong>Количество соседей = 21</strong>

# In[26]:


y_test_predict = fitKnn(21)
printKnnResult()
printError()


# <h4>Вывод на основе матриц рассогласования</h4>
# <p>Обучение модели проведено 4 раза с разным количеством <strong>соседей</strong>.</p>
# <p>Наименьшее количество ошибок 1 рода и наибольшее количество ошибок 2 рода содержится в модели с 2 <strong>соседями</strong>.</p>
# <p>Наибольшее количество ошибок 1 рода и наименьшее количество ошибок 2 рода содержится в модели с 9 <strong>соседями</strong>.</p>

# <h2>Вывод</h2>

# <p>Поставлена задача классификации с двумя классами для прогнозирования оттока клиентов банка.</p> 
# <p>Представлена визуализация и описательная статистика для данных. Рассчитана матрица корреляции и представлена ее визуализация.</p>
# <p>Проведена проверка пропущенных значений и обработка категориальных признаков. Для бинарных категориальных признаков использован метод бинаризации, а для не бинарных категориальных признаков использован метод векторизации.</p> 
# <p>Приведена причина по которой не используется нормализация количественных признаков.</p> 
# <p>Проведено разбиение данных на обучающую и тестовую выборки.</p> 
# <p>Для обучения модели используется <strong>метод k-ближайших соседей</strong>.</p> 
# <p>Проведены опыты с разным количеством <strong>соседей</strong>.</p> 
# <p>Можно сделать вывод - чем большее количество <strong>соседей</strong> используется при обучении модели, тем лучшие результаты показывает предсказание.</p>
