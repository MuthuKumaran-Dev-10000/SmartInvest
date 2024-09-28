from flask import Flask, jsonify, request
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Fetch stock data using yfinance
def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="3mo")
    return hist

# Analyze sentiment for news headlines
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    if avg_sentiment > 0.05:
        return "positive"
    elif avg_sentiment < -0.05:
        return "negative"
    else:
        return "neutral"

# Generate dynamic responses based on stock data and sentiment
def generate_response(stock_data, sentiment, stock_symbol):
    current_price = stock_data['Close'].iloc[-1]
    price_change = current_price - stock_data['Close'].iloc[0]
    trend_direction = "upward" if price_change > 0 else "downward" if price_change < 0 else "stable"

    sentiment_feedback = {
        "positive": "The recent news sentiment is positive, which could indicate further growth.",
        "negative": "The recent news sentiment is negative, so the stock might continue to decline.",
        "neutral": "The news sentiment is neutral, suggesting the stock could go either way."
    }

    response = f"[{stock_symbol}] The stock is trending {trend_direction}. Current price: ${current_price:.2f}."
    response += f" {sentiment_feedback[sentiment]}"
    return response

# API to get current stock values
@app.route('/api/stock/<string:stock_symbol>', methods=['GET'])
def get_stock_value(stock_symbol):
    stock_data = fetch_stock_data(stock_symbol)
    
    if not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        return jsonify({"stock_symbol": stock_symbol, "current_price": current_price}), 200
    else:
        return jsonify({"error": "Stock symbol not found"}), 404

# API for the chatbot response
@app.route('/api/chat', methods=['POST'])
def get_chatbot_response():
    data = request.json
    stock_symbol = data.get('stock_symbol')
    
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    stock_data = fetch_stock_data(stock_symbol)

    if not stock_data.empty:
        headlines = [
            f"{stock_symbol} has a great quarterly performance!",
            f"{stock_symbol} is facing challenges with supply chain issues.",
            f"Analysts recommend buying shares of {stock_symbol}."
        ]
        sentiment = analyze_sentiment(headlines)
        response = generate_response(stock_data, sentiment, stock_symbol)  # Pass stock_symbol to the response generator
        return jsonify({"response": response}), 200
    else:
        return jsonify({"error": "Stock symbol not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
