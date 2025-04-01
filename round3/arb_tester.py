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

# time_prices['position'] = -np.sign(time_prices['edge']) * np.log(np.abs(time_prices['edge']))/np.log(1.1)
# time_prices['position'] = np.clip(-time_prices['edge']/200*60, -60, 60)
time_prices['position'] = np.where(time_prices['edge'] > 0, -60, 60)
time_prices['position'].hist()


# %%
# the position is in the "edge" product: edge = BASKET - EQUITIES - 400
time_prices['pnl'] = time_prices['position'] * (time_prices['edge'].shift(-1) - time_prices['edge'])
time_prices['fees'] = np.abs(time_prices['position']-time_prices['position'].shift(1))*5

# plot pnl, edge, and position
fig, axs = plt.subplots(4, 1, figsize=(10, 15))
axs[0].plot(time_prices['pnl'].cumsum())
axs[0].set_title('PnL')
axs[1].plot(time_prices['edge'])
axs[1].set_title('Edge')
axs[2].plot(time_prices['position'])
axs[2].set_title('Position')
axs[3].plot(time_prices['fees'].cumsum())
axs[3].set_title('Fees')
plt.show()

time_prices
# number of 

# %%
# (time_prices['edge']/200*60).hist()

-np.sign(time_prices['edge']) #* time_prices['edge']/200*60