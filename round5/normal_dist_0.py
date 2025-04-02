# %%
import numpy as np
import matplotlib.pyplot as plt

def simulate_price_paths(n_trials, n_ticks, b=-50):
    """
    Simulate price paths starting at 0 with normal distribution steps.
    Returns the fraction of paths that go above 0 at any point.
    """
    # Generate all random walks at once (n_trials x n_ticks)
    steps = np.random.normal(0, 2.7, (n_trials, n_ticks))
    
    # Calculate cumulative sums for each trial
    paths = np.cumsum(steps, axis=1)
    
    # Check if each path ever goes above 0
    max_values = np.max(paths, axis=1)
    probability = np.mean(max_values > -b)
    
    return probability

def calculate_and_plot_probabilities():
    n_trials = 1000  # Large number of trials for low variance
    tick_ranges = np.arange(1000, 10001, 1000)  # Test from 5 to 100 ticks in steps of 5
    probabilities = []
    
    for n_ticks in tick_ranges:
        prob = simulate_price_paths(n_trials, n_ticks)
        probabilities.append(prob)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(tick_ranges, probabilities, '-o')
    plt.xlabel('Number of Ticks')
    plt.ylabel('Probability of Going Above 0')
    plt.title('Probability of Price Exceeding 0 vs Number of Ticks')
    plt.grid(True)
    plt.savefig('price_probability.png')
    plt.show()


np.random.seed(42)  # For reproducibility
calculate_and_plot_probabilities()
