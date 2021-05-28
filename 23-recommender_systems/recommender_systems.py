import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

columns_name = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('data/u.data', sep='\t', names=columns_name)
print(df.head())

movie_titles = pd.read_csv('data/Movie_Id_Titles')

print(movie_titles.head())

df = pd.merge(df, movie_titles, on='item_id')
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

sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
plt.show()

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat.head())

starwars_user_rating = moviemat['Star Wars (1977)']
liarliar_user_rating = moviemat['Liar Liar (1997)']

print(starwars_user_rating)
print(liarliar_user_rating)

similar_to_startwars = moviemat.corrwith(starwars_user_rating)
similar_to_liarliar = moviemat.corrwith(liarliar_user_rating)

corr_starwars = pd.DataFrame(similar_to_startwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False))

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending=False))
