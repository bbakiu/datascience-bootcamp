import string

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

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
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


df['tokenize'] = df['message'].apply(text_process)
print(df.head())

bow_transformer = CountVectorizer(analyzer=text_process).fit(df['message'])

print(len(bow_transformer.vocabulary_))
mess4 = df['message'][3]
print(mess4)

bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)

bow_transformer.get_feature_names()[4068]
bow_transformer.get_feature_names()[9554]

