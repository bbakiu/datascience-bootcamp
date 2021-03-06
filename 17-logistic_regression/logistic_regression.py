import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv("data/titanic_train.csv")
print(train.head())
print(train.info())

print(train.isnull())
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=train)
plt.show()

sns.displot(train['Age'].dropna(), bins=30)
plt.show()

train['Age'].plot.hist(bins=30)
plt.show()

sns.countplot(x='SibSp', data=train)
plt.show()

train['Fare'].plot.hist(bins=40, figsize=(10, 4))
plt.show()

sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()


def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, sex, embark], axis=1)
print(train.info())

train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print(train.info())

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)
print(predictions)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
