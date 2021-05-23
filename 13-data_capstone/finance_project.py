import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_datareader import data

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

# Bank of America
BAC = data.DataReader("BAC", 'stooq', start, end)

# CitiGroup
C = data.DataReader("C", 'stooq', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'stooq', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'stooq', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'stooq', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'stooq', start, end)

df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'], 'stooq', start, end)

print(df.head())
print(df.info())

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)

bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

print(bank_stocks.head())

bank_stocks.xs(key='Close', axis=1, level='Stock Info').max()

returns = pd.DataFrame()

for tick in tickers:
    returns[tick + ' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()

sns.pairplot(returns[1:])
plt.show()

print(returns.idxmin())
print(returns.idxmax())
print(returns.std())
print(returns.loc['2015-01-01':'2015-12-31'].std())

sns.histplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'], color='green', bins=100)

plt.show()

for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12, 4), label=tick)
plt.legend()
plt.show()

bank_stocks.xs(key='Close', axis=1, level='Stock Info').plot()
plt.show()


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()
plt.show()