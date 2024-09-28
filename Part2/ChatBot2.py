import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

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

# Chatbot interface
def chatbot_interface():
    print("Welcome to the AI-powered Stock Advisor Chatbot!")
    print("Ask me about stock trends and recent news.\n")

    while True:
        user_input = input("You: ").lower().strip()

        if user_input in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        if "stock" in user_input:
            stock_symbol = input("Please enter the stock symbol (e.g., AAPL): ").upper()
            stock_data = fetch_stock_data(stock_symbol)

            if not stock_data.empty:
                headlines = [
                    f"{stock_symbol} has a great quarterly performance!",
                    f"{stock_symbol} is facing challenges with supply chain issues.",
                    f"Analysts recommend buying shares of {stock_symbol}."
                ]
                sentiment = analyze_sentiment(headlines)
                response = generate_response(stock_data, sentiment)
                print(f"Chatbot: {response}")
            else:
                print(f"Chatbot: Sorry, I couldn't retrieve data for the stock symbol '{stock_symbol}'. Please check the symbol and try again.")
        else:
            print("Chatbot: I can help with stock trends and recent news. Ask about a stock!")

# Main function to run the chatbot
if __name__ == "__main__":
    chatbot_interface()
