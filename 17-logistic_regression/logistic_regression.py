import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train = pd.read_csv("titanic_train.csv")
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
