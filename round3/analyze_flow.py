# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data


# %%
prices, trades = load_round_data(3)
print("before len:", len(prices))
prices = prices[prices['overall_time'] < 2*1e6]
print("after len:", len(prices))

# %%
# Bid Volume 1

for bid_level in range(1, 4):
    fig, axs = plt.subplots(len(prices['product'].unique()), 1, figsize=(10, 10))

    for i, symbol in enumerate(prices['product'].unique()):
        sns.histplot(prices[prices['product'] == symbol][f'bid_volume_{bid_level}'], ax=axs[i])
        axs[i].set_title(f"{symbol} bid volume {bid_level}")
    plt.tight_layout()
    plt.show()

# %%
def plot_market_participants(prices, symbol, bid_volume_level):
    prices = prices[prices['product'] == symbol]
    # prices = prices[prices['overall_time'] < 10000]
    prices['best_high_vol_bid'] = np.where(prices['bid_volume_1'] > bid_volume_level, prices['bid_price_1'],
        np.where(prices['bid_volume_2'] > bid_volume_level, prices['bid_price_2'],
            np.where(prices['bid_volume_3'] > bid_volume_level, prices['bid_price_3'],
                 None)))
    
    prices['best_high_vol_ask'] = np.where(prices['ask_volume_1'] > bid_volume_level, prices['ask_price_1'],
        np.where(prices['ask_volume_2'] > bid_volume_level, prices['ask_price_2'],
            np.where(prices['ask_volume_3'] > bid_volume_level, prices['ask_price_3'],
                 None)))
    
    prices['best_low_vol_bid'] = np.where(prices['bid_volume_1'] < bid_volume_level, prices['bid_price_1'],
            np.where(prices['bid_volume_2'] < bid_volume_level, prices['bid_price_2'],
                np.where(prices['bid_volume_3'] < bid_volume_level, prices['bid_price_3'],
                    None)))
    prices['best_low_vol_ask'] = np.where(prices['ask_volume_1'] < bid_volume_level, prices['ask_price_1'],
            np.where(prices['ask_volume_2'] < bid_volume_level, prices['ask_price_2'],
                np.where(prices['ask_volume_3'] < bid_volume_level, prices['ask_price_3'],
                    None)))
    
    # add labels to the plot
    sns.lineplot(x='overall_time', y='best_high_vol_bid', data=prices, color='red')
    sns.lineplot(x='overall_time', y='best_high_vol_ask', data=prices, color='blue')
    sns.lineplot(x='overall_time', y='best_low_vol_bid', data=prices, color='green')
    sns.lineplot(x='overall_time', y='best_low_vol_ask', data=prices, color='orange')
    plt.show()

    return prices.copy()

chocolate_prices = plot_market_participants(prices, 'GIFT_BASKET', 3)

# %%
chocolate_prices['high_vol_spread'] = chocolate_prices['best_high_vol_ask'] - chocolate_prices['best_high_vol_bid']
chocolate_prices['low_vol_spread'] = chocolate_prices['best_low_vol_ask'] - chocolate_prices['best_low_vol_bid']
    
# %%
chocolate_prices['high_vol_spread'].value_counts()
chocolate_prices['low_vol_spread'].value_counts()

# %%
prices['spread'] = prices['bid_price_1'] - prices['ask_price_1']

prices.groupby('product')['spread'].mean()
    
    
    

# %%
