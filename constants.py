# Define number of tickers in universe
N_ASSETS = 10

# Define number of trading days in a year
N_TRADING_DAYS = 252

# Define confidence interval for statistics
CONFIDENCE_INTERVAL = 0.95

# Number of periods to shift returns: we get signal at day t; enter position at day (t + 1) and exit it at day (t + 2) getting return at (t + 2)
SHIFT_RETURNS = 2
