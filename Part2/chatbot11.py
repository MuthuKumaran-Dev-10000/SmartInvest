import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import dill

# Fetch stock data using yfinance
def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="3mo")
    return hist

# Analyze sentiment for news headlines (mocked for simplicity)
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
def generate_response(stock_data, sentiment):
    current_price = stock_data['Close'].iloc[-1]
    price_change = current_price - stock_data['Close'].iloc[0]
    trend_direction = "upward" if price_change > 0 else "downward" if price_change < 0 else "stable"

    sentiment_feedback = {
        "positive": "The recent news sentiment is positive, which could indicate further growth.",
        "negative": "The recent news sentiment is negative, so the stock might continue to decline.",
        "neutral": "The news sentiment is neutral, suggesting the stock could go either way."
    }

    response = f"The stock is trending {trend_direction}. Current price: ${current_price:.2f}."
    response += f" {sentiment_feedback[sentiment]}"
    return response

# Save functions to a pickle file
with open('stock_functions.pkl', 'wb') as f:
    dill.dump({
        "fetch_stock_data": fetch_stock_data,
        "analyze_sentiment": analyze_sentiment,
        "generate_response": generate_response
    }, f)
