import pandas as pd

sal = pd.read_csv('data/Salaries.csv')
print(sal.head())
print(sal.info())

print(sal['BasePay'].mean())

print(sal['OvertimePay'].max())

print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'])
print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPay'])

print(sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['EmployeeName'])

print(sal.groupby('Year').mean()['BasePay'])

print(sal['JobTitle'].nunique())

print(sal['JobTitle'].value_counts().head(5))

df2 = sal.groupby('JobTitle').count()['EmployeeName']


