# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data


# %%
prices, trades = load_round_data(3)
prices = prices[prices['overall_time'] < 1*1e6]
trades = trades[trades['overall_time'] < 1*1e6]

# %%
trades

# %%
# graph the bid_price_1 and ask_price_1 of each symbol
for symbol in prices['product'].unique():
    time_start = 0
    time_limit = 100000

    symbol_prices = prices[(prices['product'] == symbol) & (prices['overall_time'] <= time_limit) & (prices['overall_time'] >= time_start)]
    sns.lineplot(x='overall_time', y='bid_price_1', data=symbol_prices)
    sns.lineplot(x='overall_time', y='ask_price_1', data=symbol_prices)

    # add markers where trades occur
    trades_symbol = trades[(trades['symbol'] == symbol) & (trades['overall_time'] <= time_limit) & (trades['overall_time'] >= time_start)]
    plt.scatter(trades_symbol['overall_time'], trades_symbol['price'], color='red', marker='x')

    plt.title(symbol)
    plt.show()

# %%
trades.pivot_table(index='overall_time', columns='symbol', values='price')
prices.pivot_table(index='overall_time', columns='product', values='mid_price')

# %%
# check potential profit off of trades
combined = prices.merge(trades[['symbol', 'price', 'quantity', 'overall_time']], left_on=['product', 'overall_time'], right_on=['symbol', 'overall_time'], how='left')

for symbol in combined['symbol'].unique():
    symbol_combined = combined[combined['symbol'] == symbol]
    profit = np.abs(symbol_combined['price'] - symbol_combined['mid_price']) * symbol_combined['quantity']

    print(f"{symbol}: {profit.sum()}")
    print("average spread:", (symbol_combined['ask_price_1'] - symbol_combined['bid_price_1']).mean())

# %%
