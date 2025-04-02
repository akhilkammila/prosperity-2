import numpy as np
from scipy.stats import norm
import math
def call_option_value(current_price, strike=10000, ticks=250*10000):
    sigma = math.sqrt(ticks)  # Standard deviation after 250 ticks
    d = (current_price - strike) / sigma
    option_price = (current_price - strike) * norm.cdf(d) + sigma * norm.pdf(d)
    return option_price