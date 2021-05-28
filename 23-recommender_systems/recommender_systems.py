import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

columns_name = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('u.data',sep='\t', names=columns_name)
print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')

print(movie_titles.head())

df=pd.merge(df, movie_titles, on='item_id')
print(df.head())


sns.set_style('white')
print(df.groupby('title')['rating'].mean().sort_values(ascending=False))
print(df.groupby('title')['rating'].count().sort_values(ascending=False))

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())

sns.histplot(data=ratings, x='num of ratings', bins=70)
plt.show()

sns.histplot(data=ratings, x='rating', bins=50)
plt.show()

sns.jointplot(x='rating',y='num of ratings',data=ratings , alpha=0.5)
plt.show()