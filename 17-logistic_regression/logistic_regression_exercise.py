import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
train = pd.read_csv('data/advertising.csv')
print(train.head())
print(train.info)
print(train.describe())

print(train.isnull())
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

train['Age'].plot.hist(bins=30)
plt.show()
print(train.columns)

sns.jointplot(y='Area Income', x='Age', data=train)
plt.show()

sns.jointplot(y='Daily Time Spent on Site', x='Age', data=train,kind='kde',color='red' )
plt.show()


sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=train)
plt.show()

sns.pairplot(data=train, hue='Clicked on Ad')
plt.show()
X = train[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = train['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))