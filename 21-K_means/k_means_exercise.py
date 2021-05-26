import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("College_Data", index_col=0)

print(df.head())
print(df.info())
print(df.describe())

sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private',
           palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()

sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private',
           palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()

g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
plt.show()

g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()

print(df[df['Grad.Rate'] > 100])
df.loc[df['Grad.Rate'] > 100, 'Grad.Rate'] = 100
print(df[df['Grad.Rate'] > 100])

data = df.drop('Private', axis=1)
y = df['Private']

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

print(kmeans.cluster_centers_)
print(kmeans.labels_)


def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(converter)

print(df.head())

print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))
