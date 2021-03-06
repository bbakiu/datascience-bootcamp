import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

data_info = pd.read_csv('data/lending_club_info.csv', index_col='LoanStatNew')

print(data_info.head())
print(data_info.info())
print(data_info.describe())


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


feat_info('mort_acc')

df = pd.read_csv('data/lending_club_loan_two.csv')
pd.DataFrame

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

sns.countplot(x='loan_status', data=df)
plt.show()
sns.histplot(x='loan_amnt', data=df, bins=40)
plt.show()

num_df = df[['loan_amnt', 'int_rate', 'installment', 'annual_inc',
             'dti', 'open_acc', 'pub_rec', 'revol_bal',
             'revol_util', 'total_acc', 'pub_rec_bankruptcies']]

print(num_df.corr())

plt.figure(figsize=(12, 12))
sns.heatmap(num_df.corr(), annot=True, cmap='viridis')
plt.show()

feat_info('installment')
feat_info('loan_amnt')

sns.scatterplot(data=df, x='installment', y='loan_amnt')
plt.show()

sns.boxplot(data=df, x='loan_status', y='loan_amnt')
plt.show()

print(df.groupby('loan_status')['loan_amnt'].describe())

print(sorted(df['sub_grade'].unique()))
print(sorted(df['grade'].unique()))
sub_grade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='grade', hue='loan_status', data=df)
plt.show()

sns.countplot(x='sub_grade', hue='loan_status', order=sub_grade_order, data=df, palette='pastel')
plt.show()

filtered_sub_grade = filter(lambda sub_grade: (sub_grade.startswith('F')) or (sub_grade.startswith('G')),
                            sub_grade_order)

sns.countplot(x='sub_grade', hue='loan_status', order=filtered_sub_grade,
              data=df[(df['grade'] == 'F') | (df['grade'] == 'G')], palette='pastel')
plt.show()


def get_loan_repaid(status):
    if status == 'Fully Paid':
        return 1
    return 0


df['loan_repaid'] = df['loan_status'].apply(get_loan_repaid)
# df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
print(df.head())

df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()

print(df.shape[0])

print(df.isnull().sum())

null_series = df.isnull().sum() / len(df) * 100

print(null_series)

print(feat_info('emp_title'))
print(feat_info('emp_length'))

print(df['emp_title'].nunique())
print(df['emp_title'].value_counts())

df.drop('emp_title', inplace=True, axis=1)

# print(sorted(df['emp_length'].unique()))
emp_length = sorted(df['emp_length'].dropna().unique())
emp_length_order = ['< 1 year',
                    '1 year',
                    '2 years',
                    '3 years',
                    '4 years',
                    '5 years',
                    '6 years',
                    '7 years',
                    '8 years',
                    '9 years',
                    '10+ years']
print(emp_length)

sns.countplot(x='emp_length', data=df, order=emp_length_order, palette='pastel')
plt.show()

sns.countplot(x='emp_length', hue='loan_status', data=df, order=emp_length_order, palette='pastel')
plt.show()

emp_co = df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status'] == "Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co / emp_fp
print(emp_len)
emp_len.plot(kind='bar')
plt.show()
df.drop('emp_length', axis=1, inplace=True)

print(df.isnull().sum())
print(df['title'].unique())
print(df['purpose'].unique())

df.drop('title', axis=1, inplace=True)

print(feat_info('mort_acc'))
print(df['mort_acc'].value_counts())

print(df.corr()['mort_acc'])
print(df.groupby('total_acc').mean()['mort_acc'])
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
print(df.isnull().sum())

df = df.dropna()
print(df.isnull().sum())

g = df.columns.to_series().groupby(df.dtypes).groups
print(g)
print(df.select_dtypes(['object']).columns)

df['term'] = df['term'].map({'36 months': 301247, '60 months': 93972})

# grade = pd.get_dummies(df['grade'], drop_first=True)
df.drop('grade', axis=1, inplace=True)
subgrade = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df, subgrade], axis=1)

dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                         drop_first=True)
print(dummies)
df = pd.concat([df, dummies], axis=1)
df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose', 'sub_grade'], axis=1,
        inplace=True)

print(df['home_ownership'].value_counts())
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = df.drop('home_ownership', axis=1)
df = pd.concat([df, dummies], axis=1)
df['zip_code'] = df['address'].apply(lambda address: address[-5:])

dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = df.drop(['zip_code', 'address'], axis=1)
df = pd.concat([df, dummies], axis=1)

df.drop('issue_d', axis=1, inplace=True)

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df.drop('earliest_cr_line', axis=1, inplace=True)

print(df.select_dtypes(['object']).columns)

df.drop('loan_status', axis=1, inplace=True)

X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

df = df.sample(frac=0.1, random_state=101)
print(len(df))
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

# early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=3, patience=25)
history = model.fit(x=X_train, y=y_train, epochs=100, batch_size=256, validation_data=(X_test, y_test), verbose=3)

losses = pd.DataFrame(history.history)
print(losses)
losses[['loss', 'val_loss']].plot()
plt.show()

model.save('data/keras-model.h5')

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
