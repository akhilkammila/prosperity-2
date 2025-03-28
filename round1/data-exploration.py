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

# Display basic information about the loaded datasets
print("\nDataset shapes:")
print(f"Prices: {prices.shape}")
print(f"Trades: {trades.shape}")

# %%
# Plot mid prices
plot_mid_prices(prices, round)

# %%
# Analyze Starfruit returns
starfruit_returns = calculate_returns(prices, 'STARFRUIT')
analyze_returns(starfruit_returns, 'STARFRUIT')

# %%
# Percent vs. absolute return
import statsmodels.api as sm

# Create a new DataFrame with the relevant columns
returns_df = starfruit_returns[['price_difference', 'prev_price_difference']].fillna(0)

model = sm.OLS(returns_df['price_difference'], sm.add_constant(returns_df['prev_price_difference']))
results = model.fit()

# Print the summary of the regression results
print(results.summary())

# %%
# Volume graphs
prices.query('product == "STARFRUIT"')['bid_volume_1'].hist()
prices.query('product == "STARFRUIT"')['ask_volume_1'].hist()
plt.show()
prices.query('product == "STARFRUIT"')['bid_volume_2'].hist()
prices.query('product == "STARFRUIT"')['ask_volume_2'].hist()
plt.show()

prices.query('product == "STARFRUIT"')['ask_volume_3'].hist()
plt.show()

# %%
# Filter for STARFRUIT
starfruit_data = prices.query('product == "STARFRUIT" and overall_time < -1950000').copy()

# Create figure
plt.figure(figsize=(15, 8))

# For all levels
for level in [1, 2, 3]:
    # High volume participant
    high_vol_mask = starfruit_data[f'bid_volume_{level}'] > 15
    plt.scatter(starfruit_data[high_vol_mask]['overall_time'], 
               starfruit_data[high_vol_mask][f'bid_price_{level}'], 
               color='red', alpha=0.5, label='High Volume Bids' if level==1 else "")
    
    high_vol_mask = starfruit_data[f'ask_volume_{level}'] > 15
    plt.scatter(starfruit_data[high_vol_mask]['overall_time'], 
               starfruit_data[high_vol_mask][f'ask_price_{level}'], 
               color='blue', alpha=0.5, label='High Volume Asks' if level==1 else "")

    # Low volume participant
    low_vol_mask = starfruit_data[f'bid_volume_{level}'] <= 15
    plt.scatter(starfruit_data[low_vol_mask]['overall_time'], 
               starfruit_data[low_vol_mask][f'bid_price_{level}'], 
               color='magenta', alpha=0.5, label='Low Volume Bids' if level==1 else "")
    
    low_vol_mask = starfruit_data[f'ask_volume_{level}'] <= 15
    plt.scatter(starfruit_data[low_vol_mask]['overall_time'], 
               starfruit_data[low_vol_mask][f'ask_price_{level}'], 
               color='cyan', alpha=0.5, label='Low Volume Asks' if level==1 else "")

plt.title('STARFRUIT Bid/Ask Prices by Participant Type')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# High volume price
# Get the tightest price, which is high volume
starfruit_prices = prices.query('product == "STARFRUIT"').copy()

high_vol_bids = []
low_vol_bids = []
high_vol_asks = []
low_vol_asks = []
for i in range(1, 4):
    high_vol_bid = np.where(starfruit_prices[f'bid_volume_{i}'] > 15, starfruit_prices[f'bid_price_{i}'], np.nan)
    low_vol_bid = np.where(starfruit_prices[f'bid_volume_{i}'] <= 15, starfruit_prices[f'bid_price_{i}'], np.nan)
    high_vol_ask = np.where(starfruit_prices[f'ask_volume_{i}'] > 15, starfruit_prices[f'ask_price_{i}'], np.nan)
    low_vol_ask = np.where(starfruit_prices[f'ask_volume_{i}'] <= 15, starfruit_prices[f'ask_price_{i}'], np.nan)

    high_vol_bids.append(high_vol_bid)
    low_vol_bids.append(low_vol_bid)
    high_vol_asks.append(high_vol_ask)
    low_vol_asks.append(low_vol_ask)

starfruit_prices['high_vol_bid'] = np.nanmax(np.array(high_vol_bids), axis=0)
starfruit_prices['low_vol_bid'] = np.nanmax(np.array(low_vol_bids), axis=0)
starfruit_prices['high_vol_ask'] = np.nanmin(np.array(high_vol_asks), axis=0)
starfruit_prices['low_vol_ask'] = np.nanmin(np.array(low_vol_asks), axis=0)

# %%
starfruit_prices[['high_vol_bid', 'low_vol_bid', 'high_vol_ask', 'low_vol_ask']].plot()
plt.show()

# %%
for col in ['high_vol_bid', 'low_vol_bid', 'high_vol_ask', 'low_vol_ask']:
    print(col)
    print(starfruit_prices[col].notna().value_counts())

# %%
# Check the rows where there is no high_vol_bid
starfruit_prices[starfruit_prices['high_vol_bid'].isna()]

# %%
# When both high and low volume are present, what proportion is low_volume tighter
(starfruit_prices['low_vol_bid'] - starfruit_prices['high_vol_bid']).value_counts()
(starfruit_prices['low_vol_ask'] - starfruit_prices['high_vol_ask']).value_counts()

# %%
# Does the high volume price have any trends
starfruit_prices[['high_vol_bid', 'high_vol_ask']].plot()
starfruit_prices['high_vol_mid'] = (starfruit_prices['high_vol_bid'] + starfruit_prices['high_vol_ask']) / 2
high_vol_mid = ((starfruit_prices['high_vol_bid'] + starfruit_prices['high_vol_ask']) / 2).reset_index(drop=True)


plot_acf(high_vol_mid.dropna(), lags=10)
plt.show()
plot_pacf(high_vol_mid.dropna(), lags=10)
plt.show()

# %%
# Now check the returns
high_vol_returns = high_vol_mid.diff()
plot_acf(high_vol_returns.dropna(), lags=10)
plt.show()
plot_pacf(high_vol_returns.dropna(), lags=10)
plt.show()


# %%
# High volume spread
starfruit_prices['high_vol_spread'] = starfruit_prices['high_vol_ask'] - starfruit_prices['high_vol_bid']
starfruit_prices['high_vol_spread'].plot()
plt.show()

# %%
# Low volume bid/ask crosses high volume mid price
starfruit_prices['low_vol_bid_cross'] = starfruit_prices['low_vol_bid'] > starfruit_prices['high_vol_mid']
starfruit_prices['low_vol_ask_cross'] = starfruit_prices['low_vol_ask'] < starfruit_prices['high_vol_mid']
print(starfruit_prices['low_vol_bid_cross'].value_counts())
print(starfruit_prices['low_vol_ask_cross'].value_counts())

# %%
# Cross profit
starfruit_prices['low_vol_bid_cross_profit'] = starfruit_prices['low_vol_bid'] - starfruit_prices['high_vol_mid']
starfruit_prices['low_vol_ask_cross_profit'] = starfruit_prices['high_vol_mid'] - starfruit_prices['low_vol_ask']

# %%
starfruit_prices['low_vol_bid_cross_profit'].hist()
plt.show()
starfruit_prices[starfruit_prices['low_vol_bid_cross_profit'] > 0]['low_vol_bid_cross_profit'].sum()

# %%
starfruit_prices['low_vol_ask_cross_profit'].hist()
plt.show()
starfruit_prices[starfruit_prices['low_vol_ask_cross_profit'] > 0]['low_vol_ask_cross_profit'].sum()

# %%
starfruit_prices[starfruit_prices['low_vol_bid_cross_profit'] > 1]

# %%
starfruit_prices.head(15)