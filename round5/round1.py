# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import special_load_round_data

# %%
%load_ext autoreload
%autoreload 2

# %%
prices, trades = special_load_round_data(3)

# %%
def investigate_person(person, product, start_tick, end_tick):
    prices_product = prices[prices['product'] == product]\
        [(prices['overall_time'] >= start_tick) & (prices['overall_time'] <= end_tick)]
    trades_person = trades[trades['symbol'] == product]\
        [(trades['buyer'] == person) | (trades['seller'] == person)]\
        [(trades['overall_time'] >= start_tick) & (trades['overall_time'] <= end_tick)]

    trades_person['is_buy'] = trades_person['buyer'] == person
    sns.scatterplot(x='overall_time', y='bid_price_1', data=prices_product, color='red')
    sns.scatterplot(x='overall_time', y='ask_price_1', data=prices_product, color='blue')
    sns.scatterplot(x='overall_time', y='price', data=trades_person, hue='is_buy', marker='x', s=100)
    plt.title(f'{person} - {product} from {start_tick} to {end_tick}')
    plt.show()

people = trades['buyer'].unique()
for person in people:
    investigate_person(person, 'CHOCOLATE', 0, 3000000)

# %%
## check how often they are aggressor
product = 'ROSES'
prices_amethysts = prices[prices['product'] == product]
trades_amethysts = trades[trades['symbol'] == product]

combined = prices_amethysts.merge(trades_amethysts[['overall_time', 'buyer', 'seller', 'price', 'quantity']], on='overall_time', how='left')
combined['aggressor'] = np.where(combined['price'] >= combined['ask_price_1'], combined['buyer'], combined['seller'])

# %%
# when the buy, how often are they aggressor
combined.groupby('buyer').agg({'aggressor': 'value_counts'})
# combined.groupby('seller').agg({'aggressor': 'value_counts'})

# %%
# find execution prices and volumes
display(combined.groupby('buyer')['price'].value_counts())
display(combined.groupby('buyer')['quantity'].value_counts())
display(combined.groupby('seller')['price'].value_counts())
display(combined.groupby('seller')['quantity'].value_counts())

# %%
combined['buyer_agressor'] = combined['buyer'] == combined['aggressor']
combined['seller_agressor'] = combined['seller'] == combined['aggressor']
display(combined.groupby('buyer')['buyer_agressor'].value_counts())
display(combined.groupby('seller')['seller_agressor'].value_counts())

# %%
def pnl_by_person(person, start_tick, end_tick):
    # Filter trades for the given time period
    person_trades = trades[(trades['overall_time'] >= start_tick) & 
                         (trades['overall_time'] <= end_tick)]
    
    # Initialize tracking dictionaries
    positions = {}  # {symbol: quantity}
    pnl = 0
    
    # Process each trade
    for _, trade in person_trades.iterrows():
        symbol = trade['symbol']
        if symbol not in positions:
            positions[symbol] = 0
            
        if trade['buyer'] == person:
            # Person is buying
            positions[symbol] += trade['quantity']
            pnl -= trade['price'] * trade['quantity']
        elif trade['seller'] == person:
            # Person is selling
            positions[symbol] -= trade['quantity']
            pnl += trade['price'] * trade['quantity']
    
    # Mark to market at the end using last mid price for each symbol
    sub_prices = prices[prices['overall_time'] <= end_tick]
    for symbol, position in positions.items():
        if position != 0:

            last_price = sub_prices[sub_prices['product'] == symbol].iloc[-1]
            mid_price = (last_price['bid_price_1'] + last_price['ask_price_1']) / 2
            pnl += position * mid_price
    
    return {
        'pnl': pnl,
        'final_positions': positions
    }

for person in people:
    print(person, pnl_by_person(person, 0, 999999))




