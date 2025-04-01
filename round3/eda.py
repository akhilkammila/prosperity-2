# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_round_data


# %%
prices, trades = load_round_data(3)

# %%
for symbol in prices['product'].unique():
    sns.lineplot(x='overall_time', y='mid_price', data=prices[prices['product'] == symbol])
    plt.title(symbol)
    plt.show()

# %%
# look at gift basket mid price LR
import statsmodels.api as sm
mid_prices = prices.pivot(index='overall_time', columns='product', values='mid_price').reset_index()

model = sm.OLS(mid_prices['GIFT_BASKET'], sm.add_constant(mid_prices.drop(columns=['overall_time', 'GIFT_BASKET'])))
results = model.fit()
print(results.summary())

# %%
# plot residuals
residuals = results.resid
sns.lineplot(x='overall_time', y=residuals, data=mid_prices)
plt.show()

# %%
# try lr on just the first 2 days
mid_prices_2d = mid_prices[mid_prices['overall_time'] < 2*1e6]
model = sm.OLS(mid_prices_2d['GIFT_BASKET'], mid_prices_2d.drop(columns=['overall_time', 'GIFT_BASKET']))

results = model.fit()
print(results.summary())

# %%
# plot residuals
residuals = results.resid
sns.lineplot(x='overall_time', y=residuals, data=mid_prices_2d)
plt.show()

# %%
actual = mid_prices_2d['GIFT_BASKET']

x = mid_prices_2d.drop(columns=['overall_time', 'GIFT_BASKET'])
coefs = [4, 1, 6]
predicted = x.dot(coefs)

# %%
sns.lineplot(x='overall_time', y=actual, data=mid_prices_2d)
sns.lineplot(x='overall_time', y=predicted, data=mid_prices_2d)
plt.show()

# %%
sns.lineplot(x='overall_time', y=actual - predicted, data=mid_prices_2d)
plt.show()

# %%
# mean reverting difference
residuals = actual - predicted
sns.lineplot(x='overall_time', y=residuals, data=mid_prices_2d)

# %%
model = sm.OLS(residuals, sm.add_constant(actual))
results = model.fit()
print(results.summary())

# %%
# Subplot plot the basket, equities, residuals, and each symbol
fig, axs = plt.subplots(6, 1, figsize=(10, 20))

sns.lineplot(x='overall_time', y=actual, data=mid_prices_2d, ax=axs[0])
axs[0].set_title('Basket')
sns.lineplot(x='overall_time', y=predicted, data=mid_prices_2d, ax=axs[1])
axs[1].set_title('Predicted')
sns.lineplot(x='overall_time', y=residuals, data=mid_prices_2d, ax=axs[2])
axs[2].set_title('Residuals')

for i, symbol in enumerate(['CHOCOLATE', 'STRAWBERRIES', 'ROSES']):
    sns.lineplot(x='overall_time', y=symbol, data=mid_prices_2d, ax=axs[i+3])
    axs[i+3].set_title(symbol)
plt.tight_layout()
plt.show()

# %%
mid_prices_2d


# %%
for symbol in prices['product'].unique():
    sns.lineplot(x='overall_time', y=residuals, data=prices[prices['product'] == symbol])
    plt.title(symbol)
    plt.show()

# %%
