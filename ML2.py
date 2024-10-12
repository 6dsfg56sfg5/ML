import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('\\ML1\\ML\\diamond.csv')
#print(df.info())

# Проверка на пропущенные значения
#print(df.isnull().sum()) # пропусков нет

# генерируем случайную последовательность
np.random.seed(42)
df_test_nan = df.copy()
df_test_nan.loc[np.random.choice(df_test_nan.index, size=5, replace=True), 'Price'] = np.nan

#print(df_test_nan.isnull().sum()) # смотрим

# Вариант 1: Удаление строк с пропусками
#df_drop = df_test_nan.dropna()
#print("\n# Удаление строк с пропусками")

# Вариант 2: Заполнение пропусков медианой
#df_median = df_test_nan.copy()
#df_median.Price = df_median.Price.fillna(df_median.Price.median())

#print("\n# Заполнение пропусков медианой")
#print(df_median.isnull().sum()) # пропусков нет

# Вариант 3: Заполнение пропусков средним
#df_mean = df_test_nan.copy()
#df_mean.Price = df_mean.Price.fillna(df_median.Price.mean())

#print("\n# Заполнение пропусков средним")

#print(df.isnull().sum()) # пропусков нет

df['IBD'] = np.where(df['Carat Weight'] >= 2, 1, 0).astype(int)
#print(df.columns)

Y = df['IBD'] # выбираем целевую переменную (категориальную)
X = df.drop('IBD', axis=1) # переменные для проверки влияния

# В моем случае я дропаю базовую переменную, а не только. Y
X = X.drop('Carat Weight', axis=1)

# Список числовых колонок для построения графиков
numeric_cols = X.select_dtypes(include=['float64', 'int64'])

#print(numeric_cols)

# Построение boxplot для каждой переменной
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='IBD', y=col, data=df)
    plt.title(f'Boxplot {col} относительно is_best_diamond')
    #plt.show()

# Построение диаграмм распределения для каждой переменной
for col in numeric_cols:
    plt.figure(figsize=(8, 7))
    sns.histplot(data=df, x=col, hue='IBD', element="step", stat="density", common_norm=False)
    plt.title(f'Распределение {col} относительно is_best_diamond')
    #plt.show()

#for col in numeric_cols:
    #print(f"Описательная статистика для {col}:\n")
    #print(df.groupby('IBD')[col].describe())
    #print("\n" + "="*50 + "\n")

#print(df[df.select_dtypes(include=[np.number]).columns].corr())


# Выбираем категориальные признаки
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

#print("Категориальные признаки:", categorical_features)
# Создаем копию данных
X_processed = X.copy()

# Применяем One-Hot Encoding
X_processed = pd.get_dummies(X_processed, columns=categorical_features, drop_first=True)

#print(X_processed)
# Выбираем числовые признаки
numeric_features = X_processed.select_dtypes(include=['float64','int64']).columns.tolist()

#print("Числовые признаки:", numeric_features)
# Инициализируем scaler
scaler = MinMaxScaler()

# Применяем нормализацию
X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
#print(X_processed)

def train_and_evaluate(X, Y):
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Инициализация модели
    model = LogisticRegression(max_iter=1000)

    # Обучение модели
    model.fit(X_train, Y_train)

    # Предсказания на обучающей выборке
    Y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)

    # Предсказания на тестовой выборке
    Y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)

    # Вывод результатов
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

    # Классификационный отчет
    print("\nКлассификационный отчет на тестовой выборке:")
    print(classification_report(Y_test, Y_test_pred))

    return model
# Обучение модели
model = train_and_evaluate(X_processed, Y)

# Получение коэффициентов модели
coefficients = pd.DataFrame({
    'Feature': X_processed.columns,
    'Coefficient': model.coef_[0]
})
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False)

#print(coefficients[['Feature', 'Coefficient']])
from sklearn.model_selection import cross_val_score

# Кросс-валидация с 5 фолдами
scores = cross_val_score(model, X_processed, Y, cv=5, scoring='accuracy')

#print(f"Средняя точность при кросс-валидации: {scores.mean():.4f}")
#print(f"Отклонение точности: {scores.std():.4f}")
X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=42, stratify=Y)


# Предсказания вероятностей для тестовой выборки
Y_test_prob = model.predict_proba(X_test)[:, 1]

# Расчет ROC-кривой
fpr, tpr, thresholds = roc_curve(Y_test, Y_test_prob)
roc_auc = auc(fpr, tpr)

# Построение графика
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
#plt.show()
