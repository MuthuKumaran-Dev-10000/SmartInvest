import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import mplfinance as mpf
import joblib  # To save the model

# Sample current portfolio for the user (to be replaced with actual user data)
current_portfolio = {
    "AAPL": 5,
    "TSLA": 3,
    "GOOG": 2
}

# Step 1: Get the list of stocks to analyze
def get_stocks_to_analyze(current_portfolio):
    # Here we assume a hypothetical list of stocks to analyze (could be fetched from a broader index)
    all_stocks = ["AAPL", "TSLA", "GOOG", "AMZN", "MSFT", "FB", "NFLX", "NVDA", "DIS", "PYPL"]
    # Remove current portfolio stocks from the analysis
    stocks_to_analyze = [stock for stock in all_stocks if stock not in current_portfolio]
    return stocks_to_analyze

# Step 2: Fetch historical data for stocks
def fetch_data(assets):
    all_data = pd.DataFrame()
    for asset in assets:
        df = yf.download(asset, period='5y', interval='1d')  # 5 years of data
        df['Symbol'] = asset
        all_data = pd.concat([all_data, df])
    all_data.reset_index(inplace=True)  # Ensure the date is reset as a column
    all_data['Date'] = pd.to_datetime(all_data['Date'])  # Ensure 'Date' is in datetime format
    all_data.set_index('Date', inplace=True)  # Set 'Date' as the index
    return all_data

# Step 3: Load data from Excel files
def load_data_from_excel(min_file, max_file):
    min_data = pd.read_excel(min_file)
    max_data = pd.read_excel(max_file)

    # Combine both datasets for better insights
    combined_data = pd.concat([min_data, max_data])
    combined_data.reset_index(drop=True, inplace=True)

    # Convert the 'Date' column to datetime
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data.set_index('Date', inplace=True)  # Set 'Date' as the index
    return combined_data

# Step 4: Prepare data for model training
def prepare_data(data):
    data['Return'] = data['Close'].pct_change()
    data['Target'] = data['Return'].shift(-1)  # Predict next day's return
    data.dropna(inplace=True)
    
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Target']
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Step 5: Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Step 6: Predict potential high-return stocks
def predict_high_return_stocks(model, stocks, historical_data):
    predictions = {}
    for stock in stocks:
        stock_data = historical_data[historical_data['Symbol'] == stock].tail(1)  # Get last available data for prediction
        if not stock_data.empty:
            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            predicted_return = model.predict(features)[0]
            predictions[stock] = predicted_return
    return predictions

# Step 7: Generate candle charts for top predictions
def plot_candle_chart(stock, data):
    stock_data = data[data['Symbol'] == stock]
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Select relevant columns
    stock_data.index = pd.to_datetime(stock_data.index)  # Ensure index is datetime
    mpf.plot(stock_data, type='candle', style='charles', title=f"{stock} Candle Chart", volume=True)

# Main flow
stocks_to_analyze = get_stocks_to_analyze(current_portfolio)

# You can either fetch data or load from Excel
# historical_data = fetch_data(stocks_to_analyze)
historical_data = load_data_from_excel('min_Dataset.xlsx', 'max_Dataset.xlsx')

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(historical_data)

# Train the model
model = train_model(X_train, y_train)

# Predict potential high-return stocks
predictions = predict_high_return_stocks(model, stocks_to_analyze, historical_data)

# Sort and get top 5 predictions
top_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]

# Display results and plot charts
for stock, outcome in top_stocks:
    print(f"Predicted return for {stock}: {outcome:.2%}")
    plot_candle_chart(stock, historical_data)

# Save the model as a pickle file
joblib.dump(model, 'stock_prediction_model.pkl')
print("Model saved as stock_prediction_model.pkl")

print("Candle charts have been plotted for the top 5 predictions.")
