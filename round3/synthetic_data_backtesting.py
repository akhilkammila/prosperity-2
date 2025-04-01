# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data

# %%
def generate_spread(all=False, n=10000, alpha=0.0036):
    target_spread_std = 6.1

    spread = [0]  # initialize spread at 0
    for t in range(1, n):
        noise = np.random.normal(0, target_spread_std)
        new_price = (1 - alpha) * spread[-1] + noise
        spread.append(round(new_price))
    spread = pd.Series(spread)
    if not all: return spread

    equity_returns = np.random.normal(0, 5.5, n)
    equity = pd.Series(np.cumsum(equity_returns), name='Equity')
    etf = equity + spread
    return spread, equity, etf


# %%
edge_available = []
bounded_edge_available = []
for i in range(100):
    alpha = 0.0036
    spread = generate_spread()
    new_spread = np.where(np.abs(spread) < 0, 0, spread)

    curr_edge = np.abs((spread * alpha)).sum()
    edge_available.append(curr_edge*60)
    curr_bounded_edge = np.abs((new_spread * alpha)).sum()
    bounded_edge_available.append(curr_bounded_edge*60)

edge_available = pd.Series(edge_available)
bounded_edge_available = pd.Series(bounded_edge_available)
# sns.histplot(edge_available)
print('edge available', edge_available.mean())
print('bounded edge available', bounded_edge_available.mean())

# %%
sns.histplot(edge_available)

# %%
def layered_position_from_spread(spread):
    conditions = [
        (spread < -200),
        (spread < -150) & (spread >= -200),
        (spread < -100) & (spread >= -150),
        (spread < -50) & (spread >= -100),
        (spread > 200),
        (spread > 150) & (spread <= 200),
        (spread > 100) & (spread <= 150),
        (spread > 50) & (spread <= 100)
    ]

    choices = [
        60,   # spread < -200: long 60
        60,   # -200 <= spread < -150: long 45
        30,   # -150 <= spread < -100: long 30
        15,   # -100 <= spread < -50: long 15
        -60,  # spread > 200: short 60
        -60,  # 150 < spread <= 200: short 45
        -30,  # 100 < spread <= 150: short 30
        -15   # 50 < spread <= 100: short 15
    ]
    suggested_position = np.select(conditions, choices, default=0)

    actual_positions = []
    prev_position = 0

    for suggested in suggested_position:
        if np.abs(suggested) < 5:
            # If the signal is neutral, liquidate.
            current_position = 0
        else:
            # Same direction as before: only update if the new position is more extreme.
            if abs(suggested) > abs(prev_position):
                current_position = suggested
            else:
                current_position = prev_position
        actual_positions.append(current_position)
        prev_position = current_position

    # Convert the results to a pandas Series
    actual_positions = pd.Series(actual_positions, index=spread.index, name='Actual_Position')
    return actual_positions

def ayush_strategy(spread):
    position_change = np.where(spread == 0, 0, np.where(spread > 0, -1, 1))
    positions = []
    prev_position = 0
    for i in range(len(position_change)):
        prev_position += position_change[i]
        prev_position = np.clip(prev_position, -60, 60)
        positions.append(prev_position)
    return pd.Series(positions, index=spread.index, name='Position')

def ayush_strategy_with_fees(spread):
    position_change = np.where(np.abs(spread) < 50, 0, np.where(spread > 0, -1, 1))
    positions = []
    prev_position = 0
    for i in range(len(position_change)):
        prev_position += position_change[i]
        prev_position = np.clip(prev_position, -60, 60)
        positions.append(prev_position)
    return pd.Series(positions, index=spread.index, name='Position')

def linear_strategy(spread):
    position = -spread /200 * 60
    return position.clip(-60, 60)

""" 
When the spread reaches some amount, we unlock a new 'max position'
- we add 1 or -1 to the position each time, up until the max
- if we are within thershold (20), we reduce the absolute position by 1 each tick
"""
def unlock_strategy(spread):
    positions = []
    prev_position = 0
    max_long = 0
    max_short = 0

    for s in spread:
        max_long = max(max_long, -s//3) if s < 0 else 0
        max_short = min(max_short, -s//3) if s > 0 else 0
        max_long = min(max_long, 60)
        max_short = max(max_short, -60)

        if np.abs(s) > 50: # add onto the position
            prev_position -= np.sign(s)
        elif 0 <= np.abs(s) <= 5: # reduce the position
            prev_position = np.sign(prev_position) * (np.abs(prev_position) - 1) if prev_position != 0 else 0
        else:
            prev_position = 0
        
        prev_position = np.clip(prev_position, max_short, max_long)
        positions.append(prev_position)
    return pd.Series(positions, index=spread.index, name='Position')

def threshold_strategy(spread, threshold=100, close_threshold=5):
    positions = []
    prev_position = 0
    for s in spread:
        if s > threshold:
            prev_position = -60
        elif s < -threshold:
            prev_position = 60
        elif np.abs(s) < close_threshold:
            prev_position = 0
        positions.append(prev_position)
    return pd.Series(positions, index=spread.index, name='Position')
            

# %%
spread = generate_spread()
backtest_df = pd.DataFrame({'spread': spread})

backtest_df['position'] = threshold_strategy(spread)
backtest_df['pnl'] = backtest_df['position'] * (backtest_df['spread'].shift(-1).fillna(0) - backtest_df['spread'])
backtest_df['fees'] = np.abs(backtest_df['position']-backtest_df['position'].shift(1).fillna(0))*10
print('pnl', backtest_df['pnl'].sum())
print('fees', backtest_df['fees'].sum())

sns.lineplot(x=backtest_df.index, y='position', data=backtest_df)
sns.lineplot(x=backtest_df.index, y='spread', data=backtest_df)
plt.show()
sns.lineplot(x=backtest_df.index, y=(backtest_df['pnl'] - backtest_df['fees']).cumsum(), data=backtest_df)
plt.show()

# %%
# many trials testing
def test_spread_strategy(strategy_function, n_trials=100, alpha=0.0036, verbose=False):
    pnl_list = []
    fees_list = []
    profit_list = []
    for i in range(n_trials):
        spread = generate_spread(all=False, n=10000, alpha=alpha)
        position = strategy_function(spread)
        pnl = position * (spread.shift(-1).fillna(0) - spread)
        pnl_list.append(pnl.sum())
        fees = np.abs(position-position.shift(1).fillna(0))*10
        fees_list.append(fees.sum())
        profit = pnl - fees
        profit_list.append(profit.sum())
    if verbose:
        print('pnl', np.mean(pnl_list))
        print('fees', np.mean(fees_list))
        print('profit', np.mean(profit_list))
        print('profit std', round(np.std(profit_list), 2))
    return np.mean(profit_list)


thresholds = np.arange(20, 101, 20)
profits = []
for alpha in [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045]:
    profits = []
    for threshold in thresholds:
        # print(f'threshold: {threshold}')
        profit = test_spread_strategy(lambda x: threshold_strategy(x, threshold, 1), n_trials=100, alpha=alpha)
        profits.append(profit)
        # print('-'*100)
    plt.bar(thresholds, profits)
    plt.title(f'alpha {alpha} pnls w/ dif. thresholds')
    plt.show()

# thresholds = [50, 75, 100]
# close_thresholds = [1, 5, 10, 20]
# for threshold in thresholds:
#     for close_threshold in close_thresholds:
#         print(f'threshold: {threshold}, close: {close_threshold}')
#         profit = test_spread_strategy(lambda x : threshold_strategy(x, threshold, close_threshold))
#         print('-'*100)


# %%
# many trials testing ON ETF
pnl_list = []
fees_list = []
profit_list = []
for i in range(100):
    spread, equity, etf = generate_spread(True)
    alpha =0.0036
    edge_available = np.abs((spread * alpha)).sum()*60
    position = ayush_strategy_with_fees(spread)
    pnl = position * (etf.shift(-1).fillna(0) - etf)
    pnl_list.append(pnl.sum())
    fees = np.abs(position-position.shift(1).fillna(0))*4
    fees_list.append(fees.sum())
    profit = pnl - fees
    profit_list.append(profit.sum())

print('edge available', edge_available)
print('pnl', np.mean(pnl_list))
print('fees', np.mean(fees_list))
print('profit', np.mean(profit_list))
print('profit std', round(np.std(profit_list), 2))