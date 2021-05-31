import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('data/kc_house_data.csv')

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

sns.displot(df['price'])
plt.show()