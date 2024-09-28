import yfinance as yf
import pandas as pd
import joblib  # To load the model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import mplfinance as mpf

# Step 1: Get user current portfolio input
def get_user_portfolio():
    portfolio = {}
    while True:
        stock = input("Enter stock symbol (or type 'done' to finish): ").upper()
        if stock.lower() == 'done':
            break
        quantity = int(input(f"Enter quantity for {stock}: "))
        portfolio[stock] = quantity
    return portfolio

# Step 2: Get the list of stocks to analyze
def get_stocks_to_analyze(current_portfolio):
    # Assume a hypothetical list of stocks to analyze (could be fetched from a broader index)
    all_stocks = [
    "AAPL",  # Apple Inc.
    "TSLA",  # Tesla Inc.
    "GOOG",  # Alphabet Inc. (Google)
    "AMZN",  # Amazon.com Inc.
    "MSFT",  # Microsoft Corp.
    "FB",    # Meta Platforms Inc. (Facebook)
    "NFLX",  # Netflix Inc.
    "NVDA",  # NVIDIA Corporation
    "DIS",   # The Walt Disney Company
    "PYPL",  # PayPal Holdings Inc.
    "INTC",  # Intel Corporation
    "CSCO",  # Cisco Systems Inc.
    "ADBE",  # Adobe Inc.
    "V",     # Visa Inc.
    "MA",    # Mastercard Inc.
    "CRM",   # Salesforce.com Inc.
    "NFLX",  # Netflix Inc.
    "PEP",   # PepsiCo Inc.
    "KO",    # The Coca-Cola Company
    "MRK",   # Merck & Co., Inc.
    "JNJ",   # Johnson & Johnson
    "PFE",   # Pfizer Inc.
    "ABT",   # Abbott Laboratories
    "T",     # AT&T Inc.
    "VZ",    # Verizon Communications Inc.
    "NKE",   # Nike Inc.
    "WMT",   # Walmart Inc.
    "AMD",   # Advanced Micro Devices, Inc.
    "BABA",  # Alibaba Group Holding Limited
    "SQ",    # Block, Inc. (formerly Square, Inc.)
    "TGT",   # Target Corporation
    "SPGI",  # S&P Global Inc.
    "COST",  # Costco Wholesale Corporation
    "XOM",   # Exxon Mobil Corporation
    "CVX",   # Chevron Corporation
    "NFLX",  # Netflix Inc.
    "DHR",   # Danaher Corporation
    "LLY",   # Eli Lilly and Company
]
    stocks_to_analyze = [stock for stock in all_stocks if stock not in current_portfolio]
    return stocks_to_analyze

# Step 3: Fetch historical data for stocks
def fetch_data(assets):
    all_data = pd.DataFrame()
    for asset in assets:
        df = yf.download(asset, period='5y', interval='1d')  # 5 years of data
        df['Symbol'] = asset
        all_data = pd.concat([all_data, df])
    all_data.reset_index(inplace=True)
    all_data['Date'] = pd.to_datetime(all_data['Date'])  # Ensure 'Date' is in datetime format
    all_data.set_index('Date', inplace=True)  # Set 'Date' as the index
    return all_data

# Step 4: Prepare data for prediction
def prepare_data(data):
    data['Return'] = data['Close'].pct_change()
    data['Target'] = data['Return'].shift(-1)  # Predict next day's return
    data.dropna(inplace=True)

    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Target']
    return features, target

# Step 5: Predict potential high-return stocks
def predict_high_return_stocks(model, stocks, historical_data):
    predictions = {}
    for stock in stocks:
        stock_data = historical_data[historical_data['Symbol'] == stock].tail(1)  # Get last available data for prediction
        if not stock_data.empty:
            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            predicted_return = model.predict(features)[0]
            predictions[stock] = predicted_return
    return predictions

# Step 6: Generate candle charts for top predictions
def plot_candle_chart(stock, data):
    stock_data = data[data['Symbol'] == stock]
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    mpf.plot(stock_data, type='candle', style='charles', title=f"{stock} Candle Chart", volume=True)

# Main flow
# Load the saved model
model = joblib.load('stock_prediction_model.pkl')
print("Model loaded from stock_prediction_model.pkl")

# Get user current portfolio
user_portfolio = get_user_portfolio()
stocks_to_analyze = get_stocks_to_analyze(user_portfolio)
historical_data = fetch_data(stocks_to_analyze)

# Prepare data for prediction
features, _ = prepare_data(historical_data)

# Predict potential high-return stocks
predictions = predict_high_return_stocks(model, stocks_to_analyze, historical_data)

# Sort and get top 5 predictions
top_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]

# Display results and plot charts
for stock, outcome in top_stocks:
    print(f"Predicted return for {stock}: {outcome:.2%}")
    plot_candle_chart(stock, historical_data)

print("Candle charts have been plotted for the top 5 predictions.")
