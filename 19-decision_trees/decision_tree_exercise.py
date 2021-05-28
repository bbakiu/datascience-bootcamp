import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data/loan_data.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

data_hist = df[['fico', 'credit.policy']]
print(data_hist)

plt.figure(figsize=(10, 6))
data_hist[data_hist['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue',
                                                        bins=30, label='Credit.Policy=1')
data_hist[data_hist['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red',
                                                        bins=30, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

plt.figure(figsize=(10, 6))
df[df['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, color='blue',
                                           bins=30, label='Credit.Policy=1')
df[df['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, color='red',
                                           bins=30, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.show()
sns.jointplot(data=df, x='fico', y='int.rate')
plt.show()

sns.lmplot(x='fico', y='int.rate', data=df, hue='credit.policy', col='not.fully.paid')
plt.show()

cat_feats = ['purpose']
final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)

X = final_data.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
print('==========')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('==========')

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
print('==========')
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
print('==========')
