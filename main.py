import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH_TO_FILE: str = 'C:\\ML1\\ML\\diamond.csv'

df = pd.read_csv(PATH_TO_FILE) # функция превращение данных из CSV в DataFrame

df.head()
df.info()
print(df.describe())
print(df.columns)
print(df.dtypes)
for column in df.columns:
    print(f'{column}: {df[column].nunique()} уникальных значений')
print(df.describe())

sns.histplot(df['Price'], kde=True)
plt.title('Распределение Price')
plt.xlabel('Price')
plt.ylabel('Частота')
plt.show()

sns.boxplot(x=df['Price'])
plt.title('Price')
plt.xlabel('Price')
plt.show()





