import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Ecommerce Customers')

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)
plt.show()

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)
plt.show()

sns.jointplot(x='Time on App', y='Length of Membership', data=df, kind="hex")
plt.show()

sns.pairplot(df)
plt.show()

sns.lmplot(x='Yearly Amount Spent', y='Length of Membership', data=df.reset_index())
plt.show()

lm = LinearRegression()
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lm.fit(X_train, y_train)

print(lm.intercept_)
print(lm.coef_)
predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

print("Mean absolute error")
print(metrics.mean_absolute_error(y_test, predictions))
print("Mean squared error")
print(metrics.mean_squared_error(y_test, predictions))
print("Square root mean squared error")
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.hist((y_test - predictions), bins=40)
plt.show()

coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])
print(coeff)
