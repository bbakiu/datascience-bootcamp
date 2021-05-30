import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
print(yelp_class.head())

X = yelp_class['text']
y = yelp_class['stars']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
print(vectorizer.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

nb = MultinomialNB()
nb.fit(X_train, y_train)

print(nb.feature_count_)

predictions = nb.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

pipeline = Pipeline([
    ('countVectorize', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
X = yelp_class['text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))