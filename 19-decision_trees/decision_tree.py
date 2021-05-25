import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('kyphosis.csv')
print(df.head())

print(df.info())

sns.pairplot(df, hue='Kyphosis')
plt.show()

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
print('==========')
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))



