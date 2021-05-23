import matplotlib.pyplot as plt
import pandas
import pandas as pd

import seaborn as sns

sns.set_style('whitegrid')

df = pandas.read_csv('911.csv')
print(df.info())
print(df.head())
print(df['zip'].value_counts().head(5))
print(df['twp'].value_counts().head(5))

print(df['title'].nunique())


def get_reason(title):
    ch = ':'
    index = title.find(ch)
    return title[:index]


df['reason'] = df['title'].apply(lambda x: get_reason(x))
sns.countplot(x='reason', data=df)
plt.show()

print(type(df['timeStamp'].iloc[0]))

df['timeStamp'] = df['timeStamp'].apply(lambda x: pd.to_datetime(x))
print(type(df['timeStamp'].iloc[0]))

df['hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['month'] = df['timeStamp'].apply(lambda x: x.month)

df['dayOfWeek'] = df['timeStamp'].apply(lambda x: x.day_name())

sns.countplot(x='dayOfWeek', data=df, hue='reason')
plt.show()

sns.countplot(x='month', data=df, hue='reason')
plt.show()

byMonth = df.groupby(['month']).count()
print(byMonth.head())

byMonth['twp'].plot()
plt.show()

sns.lmplot(x='month', y='twp', data=byMonth.reset_index())
plt.show()
df['date'] = df['timeStamp'].apply(lambda x: x.date())

byDate = df.groupby(['date']).count()
print(byDate.head())

byDate['twp'].plot()
plt.show()

df[df['reason'] == 'EMS'].groupby('date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
plt.show()
print('============')
dayHour = df.groupby(by=['dayOfWeek', 'hour']).count()['reason'].unstack()
print(dayHour.head())

sns.heatmap(data=dayHour, cmap='coolwarm')
plt.show()

sns.clustermap(dayHour, cmap='coolwarm', standard_scale=1)
plt.show()

dayMonth = df.groupby(by=['dayOfWeek', 'month']).count()['reason'].unstack()
print(dayMonth.head())

sns.heatmap(data=dayMonth, cmap='coolwarm')
plt.show()

sns.clustermap(dayMonth, cmap='coolwarm', standard_scale=1)
plt.show()
