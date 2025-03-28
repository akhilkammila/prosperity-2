import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

def plot_mid_prices(prices_df, round_num):
    """
    Create subplots showing mid price trends for each product
    
    Args:
        prices_df (pd.DataFrame): DataFrame containing price data
        round_num (int): Round number for the title
    """
    # Calculate mid price if not already present
    if 'mid_price' not in prices_df.columns:
        prices_df['mid_price'] = (prices_df['bid'] + prices_df['ask']) / 2
    
    # Create subplots
    fig, axes = plt.subplots(nrows=len(prices_df['product'].unique()), 
                            figsize=(15, 4*len(prices_df['product'].unique())),
                            sharex=True)
    
    for (product, ax) in zip(sorted(prices_df['product'].unique()), axes):
        product_data = prices_df[prices_df['product'] == product]
        ax.plot(product_data['overall_time'], product_data['mid_price'], label=product)
        ax.set_title(f'{product} Mid Price - Round {round_num}')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_returns(prices_df, product):
    """
    Calculate returns for a specific product
    
    Args:
        prices_df (pd.DataFrame): DataFrame containing price data
        product (str): Product to analyze
        
    Returns:
        pd.DataFrame: DataFrame with returns calculated
    """
    # Filter for product and sort by time
    prod_data = prices_df[prices_df['product'] == product].copy()
    prod_data = prod_data.sort_values('overall_time')
    
    # Calculate mid price and returns
    prod_data['mid_price'] = (prod_data['bid_price_1'] + prod_data['ask_price_1']) / 2
    prod_data['return'] = prod_data['mid_price'].pct_change()
    
    # Add previous return
    prod_data['prev_return'] = prod_data['return'].shift(1)

    # Add price difference
    prod_data['price_difference'] = prod_data['mid_price'].diff()
    prod_data['prev_price_difference'] = prod_data['price_difference'].shift(1)
    
    return prod_data

def analyze_returns(returns_data, product):
    """
    Analyze returns for a given product using ACF, PACF, and binned returns
    """
    # Show correlations
    correlation = returns_data.query(f'product == "{product}"')[['return', 'prev_return']].corr()
    print(correlation)
    
    # Plot ACF and PACF
    plot_acf(returns_data['return'].dropna(), lags=20)
    plt.show()
    plot_pacf(returns_data['return'].dropna(), lags=20)
    plt.show()
    
    # Bin by previous return
    returns_data['prev_return_bin'] = pd.cut(returns_data['prev_return'], bins=10)
    sns.barplot(x='prev_return_bin', y='return', data=returns_data)
    plt.xticks(rotation=90)