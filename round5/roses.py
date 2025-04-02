# Can we identify 3 volme aggressive sells

# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import special_load_round_data

# %%
prices, trades = special_load_round_data(3)

# %%
prices = prices[prices['product'] == 'ROSES']
trades_roses = trades[(trades['symbol'] == 'ROSES') & (trades['quantity'] == 3)]

# %%
trades_roses['buyer_aggressive'] = trades_roses['buyer'] != 'Vinnie'
trades_roses['buyer_aggressive'].value_counts()


# %%
trades_roses = trades[trades['symbol'] == 'ROSES']
remy_roses = trades_roses[(trades_roses['buyer'] == 'Remy') | (trades_roses['seller'] == 'Remy')]
remy_roses['buy'] = remy_roses['buyer'] == 'Remy'
remy_roses['position'] = (remy_roses['quantity'] * np.where(remy_roses['buy'], 1, -1)).cumsum()
sns.lineplot(data=remy_roses, x='overall_time', y='position')

# sns.histplot(remy_roses['overall_time'].diff())

# %%
person = 'Remy'
trades[(trades['buyer'] == person) | (trades['seller'] == person)].groupby('overall_time').size().sort_values(ascending=False)
trades[trades['overall_time'] == 1967400]

# %%
# not vinnie
product = 'CHOCOLATE'
not_vinnie = trades[(trades['buyer'] != 'Vinnie') & (trades['seller'] != 'Vinnie')]
not_vinnie_roses = not_vinnie[not_vinnie['symbol'] == product]
not_vinnie_roses['is_buy'] = not_vinnie_roses['buyer'] == 'Vladimir'
not_vinnie_roses

# %%
# plot rose prices
sns.lineplot(data=prices[prices['product'] == product], x='overall_time', y='mid_price', color='red')
# plot not vinnie timestamps
sns.scatterplot(data=not_vinnie_roses, x='overall_time', y='price', hue='is_buy', marker='x',s=500)

# %%


