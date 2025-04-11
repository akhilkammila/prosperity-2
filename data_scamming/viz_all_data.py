# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bottle_loader import load_round_data

# %%
%load_ext autoreload
%autoreload 2

# %%
# FULL PRICES DF
prices = []
for round in [1, 2, 3, 4]:
    rd_price = load_round_data(round)
    rd_price['type'] = f'round{round}_csv'
    prices.append(rd_price)

prices = pd.concat(prices)

# %%
# Prices df from Logs + Submissions
path = '../logs/'
submissions = []
for round in ['round1', 'round2', 'round3', 'round4', 'round5', 'tutorial']:
    curr_df = pd.read_csv(path + f'{round}-submission.csv', sep=';')
    curr_df['overall_time'] = curr_df['day']*1e6 + curr_df['timestamp']
    curr_df['type'] = f'{round}_submission'
    submissions.append(curr_df)

finals = []
for round in ['round1', 'round2', 'round3', 'round4', 'round5']:
    curr_df = pd.read_csv(path + f'{round}-final.csv', sep=';')
    curr_df['overall_time'] = curr_df['day'] * 1e6 + curr_df['timestamp']
    curr_df['type'] = f'{round}_final'
    finals.append(curr_df)

submissions = pd.concat(submissions)
finals = pd.concat(finals)

# %%
combined = pd.concat([prices, submissions, finals])

# %%
# 1. Visuzlize all the data
for product in combined['product'].unique():
    product_data = combined[combined['product'] == product]
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))

    non_website = product_data[~product_data['type'].str.contains('submission')]
    website = product_data[product_data['type'].str.contains('submission')]
    sns.scatterplot(data=non_website, x='overall_time', y='mid_price', color='red', ax=axs)
    sns.scatterplot(data=website, x='overall_time', y='mid_price', hue='type', ax=axs)
    plt.title(f'{product}')
    plt.show()

# %%
# Plot the scuffed data
path = './website_submissions/'
day1_data = pd.read_csv(path + 'rd1_v1.csv', sep=';')
scam_data = day1_data[day1_data['product'] == 'SQUID_INK']['mid_price'].diff().fillna(0)
scam_data = pd.Series(scam_data)

# %%
for product in submissions['product'].unique():
    temp_data = submissions[submissions['product'] == product]
    for day in temp_data['day'].unique():
        product_data = temp_data[temp_data['day'] == day]
        product_data = product_data['mid_price'].diff().fillna(0)
        product_data = pd.Series(product_data)
        # print(scam_data.head(), product_data.head())
        print('product:', product, 'day:', day)
        print('correlation:', pd.Series(scam_data.values).corr(pd.Series(product_data.values)))
        

# %%
submissions
