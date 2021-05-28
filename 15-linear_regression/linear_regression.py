import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/USA_Housing.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
sns.pairplot(df)
plt.show()
sns.displot(df['Price'])
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)
print(lm.coef_)
coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])
print(coeff)

predictions = lm.predict(X_test)
print(predictions)

plt.scatter(y_test, predictions)
plt.show()

sns.displot((y_test-predictions))
plt.show()

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))