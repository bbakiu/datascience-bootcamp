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