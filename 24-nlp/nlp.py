import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

print(df[df['length']==910]['message'].iloc[0])

df.hist(column='length', by='label', bins=60, figsize=(12,4))
plt.show()
