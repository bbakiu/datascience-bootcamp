import string

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# nltk.download_shell()
messages = [line.rstrip() for line in open('data/SMSSpamCollection')]

print(len(messages))

print(messages[50])

print(messages[0:10])

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')

df = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['label', 'message'])
print(df.head())

print(df.describe())

print(df.groupby('label').describe())

df['length'] = df['message'].apply(len)
print(df.head())
df['length'].plot.hist(bins=150)
plt.show()

print(df['length'].describe())

print(df[df['length'] == 910]['message'].iloc[0])

df.hist(column='length', by='label', bins=60, figsize=(12, 4))
plt.show()

mess = 'Sample message! Notice: it has punctuation.'

print(string.punctuation)

nopunc = [c for c in mess if c not in string.punctuation]
nopunc = ''.join(nopunc)

print(nopunc)
nopunc.split()
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print(clean_mess)


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower()
    return [word for word in nopunc.split() if word not in stopwords.words('english')]


df['tokenize'] = df['message'].apply(text_process)
print(df.head())

bow_transformer = CountVectorizer(analyzer=text_process).fit(df['message'])

print(len(bow_transformer.vocabulary_))
mess4 = df['message'][3]
print(df['tokenize'][3])
print(mess4)

bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)

# print(bow_transformer.get_feature_names()[9530])

messages_bow = bow_transformer.transform(df['message'])

print('Shape of sparse matrix', messages_bow.shape)

print(messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print(('sparsity: {}'.format(sparsity)))

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)

spam_detect_model = MultinomialNB().fit(messages_tfidf,df['label'])

print(spam_detect_model.predict(tfidf4)[0])

all_pred = spam_detect_model.predict(messages_tfidf)
print(all_pred)

msg_train, msg_test, label_train, label_test = train_test_split(df['message'], df['label'], test_size=0.3)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test, predictions))

