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
prices = prices[prices['overall_time'] < 3*1e6]
trades = trades[trades['overall_time'] < 3*1e6]

# %%
time_prices = prices.pivot_table(index='overall_time', columns='product', values='mid_price')
time_prices['EQUITIES'] = time_prices['CHOCOLATE']*4 + time_prices['ROSES']*1 + time_prices['STRAWBERRIES']*6

time_prices['edge'] = time_prices['GIFT_BASKET'] - time_prices['EQUITIES'] - 400
time_prices = time_prices.reset_index()
time_prices

# %%
# plot_acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
for product in ['CHOCOLATE', 'GIFT_BASKET', 'ROSES', 'STRAWBERRIES']:
    returns = time_prices[product].diff().fillna(0)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plot_acf(returns, ax=axs[0])
    plot_pacf(returns, ax=axs[1])
    axs[0].set_title(f'{product} ACF')
    axs[1].set_title(f'{product} PACF')
plt.show()
;

# %%
# if it goes up 2 times in a row, what happens?
returns_df = time_prices.diff().fillna(0).drop(columns=['overall_time', 'edge'])
returns_df

# %%
# histogram of returns
returns_df.hist()
plt.tight_layout()
plt.show()

# %% 
for product in returns_df.columns:
    sign = returns_df[product].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    sign_dif = (sign.shift(1) + sign.shift(2) + sign.shift(3)).fillna(0)
    returns_df.groupby(sign_dif)[product].hist(label=product)

    # sns.histplot(returns_df.groupby(sign_dif)[product])
    
    plt.show()
plt.show()

# %%
# what is the local streak?
extremes_df = pd.DataFrame()
for product in returns_df.columns:
    # Shifted columns
    prev = returns_df[product].shift(1)
    next_ = returns_df[product].shift(-1)

    # Identify local mins and maxes
    is_local_max = (returns_df[product] > prev) & (returns_df[product] > next_)
    is_local_min = (returns_df[product] < prev) & (returns_df[product] < next_)
    is_extreme = is_local_max | is_local_min

    #
    extremes = returns_df[product].where(is_extreme)
    last_extreme = extremes.ffill()

    dif_from_last_extreme = returns_df[product] - last_extreme
    sns.kdeplot(dif_from_last_extreme)
    plt.title(product)
    plt.show()

    extremes_df[product] = extremes

# %%
# big price jumps
for product in returns_df.columns:
    # plot count of each return
    returns_df[product].value_counts().sort_index().plot(kind='bar')
    plt.title(f'{product} returns')
    plt.show()

# %%
# after a large return in equities, what will the next return be?
returns_df['last_big'] = returns_df['EQUITIES'].shift(1).apply(lambda x: 1 if x > 20 else (-1 if x < -20 else 0))
returns_df.groupby('last_big')['EQUITIES'].describe()

# %%