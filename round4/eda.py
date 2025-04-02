# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data


# %%
prices, trades = load_round_data(4)
# prices = prices[prices['overall_time'] < 2e6]

# %%
for symbol in prices['product'].unique():
    sns.lineplot(x='overall_time', y='mid_price', data=prices[prices['product'] == symbol])
    plt.title(symbol)
    plt.show()

# %%
""" 
Things to test
1. are coconut returns normal
    - are they dependent on the current price or no (mean-reverting or no)
    - is it same normal std over different periods
2. are coconut returns correlated with any other products, or just random
3. do black scholes on coconut coupon
    - check mean reversion
"""

# %%
# basic histplot to see if it looks normal
coconut_prices = prices[prices['product'] == 'COCONUT']
sns.histplot(coconut_prices['mid_price'].diff().fillna(0))

# %%
# are returns autocorrelated
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(coconut_prices['mid_price'].diff().fillna(0))
plot_pacf(coconut_prices['mid_price'].diff().fillna(0))

# %%
# are returns dependent on the current price
coconut_df = pd.DataFrame({'price': coconut_prices['mid_price'], 'returns': coconut_prices['mid_price'].diff().fillna(0)})
coconut_df['price_bucket'] = pd.cut(coconut_df['price'], bins=np.arange(9800, 10201, 50))
coconut_df.groupby('price_bucket')['returns'].describe()
# %%
# check if coconut returns are normal with shapiro-wilk test
from scipy.stats import shapiro
shapiro(coconut_df['returns'])

# %%
# plot coconut returns hist over different time periods
periods = 5
fig, axs = plt.subplots(periods, 1, figsize=(10, 5*periods))
time_split = len(coconut_df) // periods


for i in range(periods):
    sub_prices = coconut_df.iloc[i*time_split:(i+1)*time_split]
    sns.histplot(sub_prices['returns'], ax=axs[i])
    axs[i].set_title(f'Time period {i+1}')
    print(sub_prices['returns'].describe())

# make plots share x axis
plt.setp(axs, xlim=(-4.5, 4.5))
plt.show()
# %%
sample_normal = np.random.normal(0, 1, 10000)
sample_normal_int = pd.Series(sample_normal.round())
# sns.histplot(sample_normal_int)
sample_normal_int.describe()
