import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import mplfinance as mpf
import joblib

# Step 1: Get user portfolio input
def get_user_portfolio():
    portfolio = {}
    while True:
        stock = input("Enter stock symbol (or type 'done' to finish): ").upper()
        if stock.lower() == 'done':
            break
        quantity = int(input(f"Enter quantity for {stock}: "))
        portfolio[stock] = quantity
    return portfolio

# Step 2: Fetch historical data for stocks
def fetch_data(assets):
    all_data = pd.DataFrame()
    for asset in assets:
        df = yf.download(asset, period='5y', interval='1d')  # 5 years of data
        df['Symbol'] = asset
        all_data = pd.concat([all_data, df])
    return all_data.reset_index()

# Step 3: Prepare data for model training
def prepare_data(data):
    data['Return'] = data['Close'].pct_change()
    data['Target'] = data['Return'].shift(-1)  # Predict next day's return
    data['SMA'] = data['Close'].rolling(window=14).mean()  # Simple Moving Average
    data['Volatility'] = data['Close'].rolling(window=14).std()  # Volatility
    data.dropna(inplace=True)
    
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'Volatility']]
    target = data['Target']
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate the model with cross-validation
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    print(f"Model accuracy: {scores.mean():.2%} +/- {scores.std():.2%}")

# Step 6: Predict and evaluate portfolio
def predict_portfolio(portfolio, model, data):
    predictions = {}
    for stock in portfolio.keys():
        stock_data = data[data['Symbol'] == stock].tail(1)  # Get last available data for prediction
        if not stock_data.empty:
            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'Volatility']].values
            predicted_return = model.predict(features)[0]
            predictions[stock] = predicted_return
    return predictions

# Step 7: Generate candle charts for top predictions
def plot_candle_chart(stock, data):
    stock_data = data[data['Symbol'] == stock]
    mpf.plot(stock_data, type='candle', style='charles', title=f"{stock} Candle Chart", volume=True)

# Step 8: Save the trained model
def save_model(model, filename='stock_prediction_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Main flow
user_portfolio = get_user_portfolio()
assets = list(user_portfolio.keys())
historical_data = fetch_data(assets)

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(historical_data)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_train, y_train)

# Save the model
save_model(model)

# Predict portfolio outcomes
predictions = predict_portfolio(user_portfolio, model, historical_data)

# Sort and get top 3 predictions
top_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

# Display results
for stock, outcome in top_stocks:
    current_price = historical_data[historical_data['Symbol'] == stock]['Close'].values[-1]
    profit = outcome * user_portfolio[stock] * current_price
    print(f"Predicted return for {stock}: {outcome:.2%} (Profit: ${profit:.2f})")
    plot_candle_chart(stock, historical_data)

print("Candle charts have been plotted for the top 3 predictions.")
