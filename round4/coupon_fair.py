# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data
from option_pricing import call_option_value

# %%
%load_ext autoreload
%autoreload 2

# %%
prices, trades = load_round_data(4)
# prices = prices[prices['overall_time'] < 2e6]

# %%
for symbol in prices['product'].unique():
    sns.lineplot(x='overall_time', y='mid_price', data=prices[prices['product'] == symbol])
    plt.title(symbol)
    plt.show()

# %%
prices_pivoted = prices.pivot(index='overall_time', columns='product', values='mid_price')
coconut_prices = prices_pivoted['COCONUT']
coconut_prices

# Create a new column for coupon prices
coupon_prices = coconut_prices.apply(lambda x: call_option_value(x))

# Plot both COCONUT and coupon prices
plt.figure(figsize=(12, 6))
plt.plot(coconut_prices.index, prices_pivoted['COCONUT_COUPON'].values, label='COUPON actual price')
plt.plot(coconut_prices.index, coupon_prices.values, label='Coupon Expected Price')
plt.title('COCONUT Price vs Coupon Price')
plt.legend()
plt.show()