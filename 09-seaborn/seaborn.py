import seaborn as sns
import matplotlib.pyplot as plt

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


