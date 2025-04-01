""" 
Trying to predict their process
    - maybe normal dist. + some drift?
    - look at the absolute difference between equity and etf,
        then see how it changes based on its current value
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data

# %%
# arb backtest
prices, trades = load_round_data(3)
prices = prices[prices['overall_time'] < 1*1e6]
trades = trades[trades['overall_time'] < 1*1e6]

# %%
time_prices = prices.pivot_table(index='overall_time', columns='product', values='mid_price')
time_prices['EQUITIES'] = time_prices['CHOCOLATE']*4 + time_prices['ROSES']*1 + time_prices['STRAWBERRIES']*6
time_prices['edge'] = time_prices['GIFT_BASKET'] - time_prices['EQUITIES'] - 400

# %%
(np.abs(time_prices['edge'])*0.0036*60).sum()

# %%
time_prices['equity_diff'] = (time_prices['EQUITIES'].shift(-1) - time_prices['EQUITIES']).fillna(0)
time_prices['basket_diff'] = (time_prices['GIFT_BASKET'].shift(-1) - time_prices['GIFT_BASKET']).fillna(0)
time_prices['edge_diff'] = (time_prices['edge'].shift(-1) - time_prices['edge']).fillna(0)

# %%
# filter by the current edge value
time_prices['edge_bucket'] = pd.cut(time_prices['edge'], bins=np.arange(-250, 251, 50))
for bucket in time_prices['edge_bucket'].unique():
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    sns.kdeplot(time_prices[time_prices['edge_bucket'] == bucket]['equity_diff'], ax=axs[0])
    sns.kdeplot(time_prices[time_prices['edge_bucket'] == bucket]['basket_diff'], ax=axs[1])
    axs[0].set_title(f'Equity diff for edge bucket {bucket}')
    axs[1].set_title(f'Basket diff for edge bucket {bucket}')

    # make the plots share x-axis
    axs[0].set_xlim(axs[0].get_xlim())
    axs[1].set_xlim(axs[0].get_xlim())
    plt.tight_layout()
    plt.show()

# %%
time_prices[['equity_diff', 'basket_diff', 'edge_diff', 'edge_bucket']].groupby('edge_bucket').agg(['mean', 'std', 'count'])

# %%
time_prices[['equity_diff', 'basket_diff', 'edge_diff']].std()

# %% 
# OLS
import statsmodels.api as sm
model = sm.OLS(time_prices['basket_diff'], time_prices['edge'])
results = model.fit()
print(results.summary())

""" 
Generating our own data
"""

# %%
np.random.seed(13)
n = 10000

# --- Simulate equity series ---
# Equity returns with std 5.5 (e.g., percentage points)
equity_returns = np.random.normal(0, 5.5, n)
equity = pd.Series(np.cumsum(equity_returns), name='Equity')

# --- Generate mean-reverting spread ---
alpha = 0.0036  # mean-reversion speed
target_spread_std = 6.1

spread = [0]  # initialize spread at 0
for t in range(1, n):
    noise = np.random.normal(0, target_spread_std)
    spread.append((1 - alpha) * spread[-1] + noise)
spread = pd.Series(spread, name='Spread')

# --- Construct ETF series as equity plus spread ---
etf = equity + spread

# plot the three series
plt.figure(figsize=(12, 8))
plt.plot(equity, label='Equity')
plt.plot(spread, label='Spread')
plt.plot(etf, label='ETF')
plt.legend()
plt.show()

# %%
# find the metrics of equity diff, spread diff, and edge diff
equity_diff = equity.diff()
spread_diff = spread.diff()
etf_diff = etf.diff()
edge_buckets = pd.cut(spread, bins=np.arange(-250, 251, 50))

synthetic_data = pd.DataFrame({
    'equity_diff': equity_diff,
    'etf_diff': etf_diff,
    'spread_diff': spread_diff,
    'edge_bucket': edge_buckets
})

synthetic_data.groupby('edge_bucket').agg(['mean', 'std', 'count'])
