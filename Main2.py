import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample current portfolio for the user
current_portfolio = {
    "AAPL": 5,
    "TSLA": 3,
    "GOOG": 2
}

# Step 1: Get the list of stocks to analyze
def get_stocks_to_analyze(current_portfolio):
    all_stocks = ["AAPL", "TSLA", "GOOG", "AMZN", "MSFT", "FB", "NFLX", "NVDA", "DIS", "PYPL"]
    stocks_to_analyze = [stock for stock in all_stocks if stock not in current_portfolio]
    return stocks_to_analyze

# Step 2: Fetch historical data for stocks
def fetch_data(assets):
    all_data = pd.DataFrame()
    for asset in assets:
        try:
            df = yf.download(asset, period='2y', interval='1d')  # 2 years of data
            if df.empty:
                print(f"No data found for {asset}. It may be delisted.")
                continue
            df['Symbol'] = asset
            all_data = pd.concat([all_data, df], ignore_index=True)  # Ignore empty entries
        except Exception as e:
            print(f"Error fetching data for {asset}: {e}")
    
    # Check if all_data is empty
    if all_data.empty:
        print("No data was fetched for any stocks.")
        return None

    all_data.reset_index(inplace=True)
    if 'Date' in all_data.columns:
        all_data['Date'] = pd.to_datetime(all_data['Date'])
        all_data.set_index('Date', inplace=True)
    else:
        print("No 'Date' column found in fetched data.")
    
    return all_data

# Step 3: Prepare data for model training
def prepare_data(data):
    data['Return'] = data['Close'].pct_change()
    data['Target'] = data['Return'].shift(-1)  # Predict next day's return
    data.dropna(inplace=True)
    
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Target']
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Step 5: Predict future stock prices for the next n days
def predict_future_prices(model, last_data, days=14):
    predictions = []
    for _ in range(days):
        features = last_data[['Open', 'High', 'Low', 'Close', 'Volume']].values[-1].reshape(1, -1)
        predicted_return = model.predict(features)[0]
        
        # Update last_data for the next prediction
        next_price = last_data['Close'].values[-1] * (1 + predicted_return)
        next_row = pd.Series({
            'Open': next_price,
            'High': next_price,
            'Low': next_price,
            'Close': next_price,
            'Volume': 0,  # Volume can be set to 0 or average historical volume
        })
        
        # Use _append instead of append
        last_data = last_data._append(next_row, ignore_index=True)
        predictions.append(next_price)
        
    return predictions

# Main flow
stocks_to_analyze = get_stocks_to_analyze(current_portfolio)
historical_data = fetch_data(stocks_to_analyze)

# Prepare data only if historical_data is not None
if historical_data is not None:
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(historical_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Get the last data point for each stock for predictions
    predictions = {}
    for stock in stocks_to_analyze:
        last_data = historical_data[historical_data['Symbol'] == stock].tail(1)  # Get last available data for prediction
        if not last_data.empty:
            future_prices = predict_future_prices(model, last_data)
            predictions[stock] = future_prices

    # Display predictions for the next 1-2 weeks
    for stock, future_prices in predictions.items():
        print(f"Predicted future prices for {stock} for the next 14 days:")
        for i, price in enumerate(future_prices, start=1):
            print(f"Day {i}: ${price:.2f}")

    # Save the model as a pickle file
    joblib.dump(model, 'stock_prediction_model.pkl')
    print("Model saved as stock_prediction_model.pkl")
else:
    print("Failed to fetch historical data for any stocks.")
