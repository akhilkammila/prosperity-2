# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_round_data
from plotting import (plot_mid_prices, calculate_returns, analyze_returns)

# %%
%load_ext autoreload
%autoreload 2

# %%
# Load data
round = 1
prices, trades = load_round_data(round)

# %%
# Plot mid prices
plot_mid_prices(prices, round)

# %%
starfruit_prices = prices[prices['product'] == 'STARFRUIT']
starfruit_prices['best_high_vol_bid'] = np.where(starfruit_prices['bid_volume_1'] > 15, starfruit_prices['bid_price_1'],
                                                np.where(starfruit_prices['bid_volume_2'] > 15, starfruit_prices['bid_price_2'],
                                                         np.where(starfruit_prices['bid_volume_3'] > 15, starfruit_prices['bid_price_3'],
                                                                  None)))
                                        
starfruit_prices['best_high_vol_ask'] = np.where(starfruit_prices['ask_volume_1'] > 15, starfruit_prices['ask_price_1'],
                                                np.where(starfruit_prices['ask_volume_2'] > 15, starfruit_prices['ask_price_2'],
                                                         np.where(starfruit_prices['ask_volume_3'] > 15, starfruit_prices['ask_price_3'],
                                                                  None)))

# %%
starfruit_prices['best_high_vol_bid'] = np.where(starfruit_prices['best_high_vol_bid'].isna(), starfruit_prices['best_high_vol_ask'] - 7, starfruit_prices['best_high_vol_bid'])
starfruit_prices['best_high_vol_ask'] = np.where(starfruit_prices['best_high_vol_ask'].isna(), starfruit_prices['best_high_vol_bid'] + 7, starfruit_prices['best_high_vol_ask'])

# %%
starfruit_prices['true_mid_price'] = (starfruit_prices['best_high_vol_bid'] + starfruit_prices['best_high_vol_ask']) / 2
starfruit_prices['true_spread'] = starfruit_prices['best_high_vol_ask'] - starfruit_prices['best_high_vol_bid']

# %%
# Orderbook money available
for i in range(1, 4):
    starfruit_prices[f'money_available_{i}'] = starfruit_prices[f'ask_volume_{i}'] * starfruit_prices['true_mid_price'] - starfruit_prices[f'ask_price_{i}'] - 

# %

# %%

