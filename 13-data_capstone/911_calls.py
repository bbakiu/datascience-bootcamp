import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy as np

import seaborn as sns
sns.set_style('whitegrid')

df = pandas.read_csv('911.csv')
print(df.info())
print(df.head())
print(df['zip'].value_counts().head(5))
