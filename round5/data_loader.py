import pandas as pd

def special_load_round_data(round_num):
    """
    Load price and trade data for a specific round (3 days before the round)
    
    Args:
        round_num (int): The round number to load data for
        
    Returns:
        tuple: (prices_df, trades_df) containing the concatenated data for all days
    """
    price_path = f"../2024_data_bottles/round-{round_num}-island-data-bottle/"
    trade_path = f"../2024_data_bottles/round-5-island-data-bottle/"


    # Create empty lists to store dataframes
    prices_dfs = []
    trades_dfs = []
    
    # Load data for each day
    for day in range(round_num-3, round_num):
        # Load prices
        price_file = f"prices_round_{round_num}_day_{day}.csv"
        prices_df = pd.read_csv(f"{price_path}{price_file}", sep=';')
        prices_df['day'] = day
        prices_dfs.append(prices_df)
    
    for day in range(round_num-3, round_num):
        # Load trades
        trade_file = f"trades_round_{round_num}_day_{day}_wn.csv"
        trades_df = pd.read_csv(f"{trade_path}{trade_file}", sep=';')
        trades_df['day'] = day
        trades_dfs.append(trades_df)
    
    # Combine all days into single dataframes
    prices = pd.concat(prices_dfs, ignore_index=True)
    trades = pd.concat(trades_dfs, ignore_index=True)
    
    # Add overall time
    prices['overall_time'] = prices['timestamp'] + prices['day'] * 1e6
    trades['overall_time'] = trades['timestamp'] + trades['day'] * 1e6
    
    return prices, trades 