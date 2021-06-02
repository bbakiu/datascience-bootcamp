import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

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

model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y = y_train, epochs=600, validation_data=(X_test, y_test))

lossed = pd.DataFrame(model.history.history)
lossed.plot()
plt.show()


model2 = Sequential()

model2.add(Dense(30, activation='relu'))
model2.add(Dense(15, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model2.fit(x=X_train, y = y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stopping])

lossed2 = pd.DataFrame(model2.history.history)
lossed2.plot()
plt.show()


model3 = Sequential()

model3.add(Dense(30, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(15, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model3.fit(x=X_train, y = y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stopping])

lossed3 = pd.DataFrame(model3.history.history)
lossed3.plot()
plt.show()

predictions = model3.predict_classes(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))