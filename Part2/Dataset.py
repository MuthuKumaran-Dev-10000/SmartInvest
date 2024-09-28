import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Create a sample dataset of stock prices
data = {
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'StockSymbol': ['AAPL'] * 100,
    'ClosePrice': np.cumsum(np.random.normal(0, 1, 100)) + 150  # Simulate a fluctuating stock price
}

df = pd.DataFrame(data)

# Display the first few rows
print(df.head())
