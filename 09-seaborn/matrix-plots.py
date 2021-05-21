import matplotlib.pyplot as plt

import seaborn as sns

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

tips.head()
flights.head()

tc = tips.corr()
sns.heatmap(tc, annot=True, cmap='coolwarm')
plt.show()

fp = flights.pivot_table(index='month', columns='year', values='passengers')
print(fp)

sns.heatmap(fp, cmap='coolwarm', linecolor='white', linewidth=3)
plt.show()

sns.clustermap(fp, cmap='coolwarm', standard_scale=1)
plt.show()
