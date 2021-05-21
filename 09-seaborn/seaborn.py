import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tips = sns.load_dataset('tips')

print(tips.head())

sns.distplot(tips['total_bill'])
plt.show()
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')
plt.show()
sns.pairplot(tips, hue="sex", palette='coolwarm')

plt.show()
sns.rugplot(tips['total_bill'])
plt.show()

sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
plt.show()

sns.countplot(x='sex', data=tips)
plt.show()

sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()

sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()

sns.stripplot(x='day', y='total_bill', data=tips)
plt.show()

sns.swarmplot(x='day', y='total_bill', data=tips)
plt.show()


sns.factorplot(x='day', y='total_bill', data=tips, kind='bar')
plt.show()
