import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/kc_house_data.csv')

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

sns.displot(df['price'])
plt.show()

sns.countplot(df['bedrooms'])
plt.show()

print(df.corr()['price'].sort_values())

sns.scatterplot(x='price', y='sqft_living', data=df)
plt.figure(figsize=(12, 6))
plt.show()

sns.boxplot(x='bedrooms', y='price', data=df)
plt.show()

sns.scatterplot(x='price', y='long', data=df)
plt.figure(figsize=(12, 8))
plt.show()

sns.scatterplot(x='price', y='lat', data=df)
plt.figure(figsize=(12, 8))
plt.show()

sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.figure(figsize=(12, 8))
plt.show()

print(df.sort_values('price', ascending=False).head(215))

bottom_99 = df.sort_values('price', ascending=False).iloc[216:]

sns.scatterplot(x='long', y='lat', data=bottom_99, hue='price', edgecolor=None, alpha=0.2, palette='RdYlGn')
plt.figure(figsize=(12, 8))
plt.show()

sns.boxplot(x='waterfront', y='price', data=df)
plt.show()

df.drop('id', axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

sns.boxplot(x='month', y='price', data=df)
plt.show()

df.groupby('month').mean()['price'].plot()
plt.show()

df.drop('date', axis=1, inplace=True)



