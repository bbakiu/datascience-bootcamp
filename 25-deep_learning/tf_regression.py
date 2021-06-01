import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

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

df['zipcode'].value_counts()

df.drop('zipcode', axis=1, inplace=True)
# df.drop('yr_renovated', axis=1, inplace=True)

X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=400, validation_data=(X_test, y_test), batch_size=128)

losses = pd.DataFrame(model.history.history)
losses.plot()

predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))

print(df['price'].describe())

print(explained_variance_score(y_test, predictions))

singlehouse = df.drop('price', axis=1).iloc[0]
singlehouse = scaler.transform(singlehouse.values.reshape(-1, 19))

print(model.predict(singlehouse))
