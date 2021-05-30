import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

yelp = pd.read_csv('data/yelp.csv')
print(yelp.head())
print(yelp.columns)
print((yelp.info()))
print((yelp.describe()))

yelp['text_length'] = yelp['text'].apply(len)

g = sns.FacetGrid(data=yelp, col='stars')
g.map_dataframe(sns.histplot, x='text_length')
g.set_axis_labels("Start", "Text length")
plt.show()

sns.boxplot(data=yelp, x='stars', y='text_length')
plt.show()

sns.countplot(data=yelp, x='stars')
plt.show()

means = yelp.groupby(['stars'])[['cool', 'useful', 'funny', 'text_length']].mean()
print(means)

print(means.corr())

sns.heatmap(data=means.corr(), cmap='coolwarm', annot=True)
plt.show()
