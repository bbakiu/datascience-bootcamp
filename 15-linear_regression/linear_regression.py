import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('USA_Housing.csv')

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
