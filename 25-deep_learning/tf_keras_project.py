import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
emp_length_order = [ '< 1 year',
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