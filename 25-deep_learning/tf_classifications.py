import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/cancer_classification.csv')

print(df.head())
print(df.info())
print(df.describe())

sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
plt.show()

sns.heatmap(df.corr())
plt.show()

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

