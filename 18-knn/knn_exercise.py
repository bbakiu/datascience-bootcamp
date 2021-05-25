import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("KNN_Project_Data")

print(df.head())
print(df.info())

sns.pairplot(df, hue='TARGET CLASS')
plt.show()

data = df.drop('TARGET CLASS', axis=1)
scaler = StandardScaler()
scaler.fit(data)
scaled_feature = scaler.transform(data)
print(scaled_feature)
print(data.columns)
df_feat = pd.DataFrame(scaled_feature, columns=data.columns)
print(df_feat.info())

X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predictions_i = knn.predict(X_test)
    error_rate.append(np.mean(predictions_i != y_test))

print(error_rate)
plt.plot(range(1, 40), error_rate, color='blue')
plt.show()

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
