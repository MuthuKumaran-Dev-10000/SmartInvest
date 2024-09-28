import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import random

# Step 1: Create a sample dataset with multiple stock prices
def create_dataset():
    stock_symbols = ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT', 'NFLX', 'FB', 'NVDA', 'INTC', 'AMD']
    stock_data = []

    for symbol in stock_symbols:
        data = {
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'StockSymbol': [symbol] * 100,
            'ClosePrice': np.cumsum(np.random.normal(0, 1, 100)) + np.random.uniform(50, 500)
        }
        stock_data.append(pd.DataFrame(data))

    return pd.concat(stock_data)

# Step 2: Calculate the simple moving average (SMA)
def calculate_sma(data, window=7):
    return data['ClosePrice'].rolling(window=window).mean()

# Step 3: Fetch news articles using web scraping
def fetch_news(stock_symbol):
    search_url = f"https://news.google.com/search?q={stock_symbol}%20stock"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract titles or snippets from the news articles
    headlines = soup.find_all('a', class_='DY5T1d')
    
    # Return the first 3 news headlines for simplicity
    return [headline.get_text() for headline in headlines[:3]]

# Step 4: Perform sentiment analysis on news headlines
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    
    # Average the sentiment scores
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    if avg_sentiment > 0.05:
        return "positive"
    elif avg_sentiment < -0.05:
        return "negative"
    else:
        return "neutral"

# Step 5: Analyze stock trends and provide a response
def analyze_stock_trend(data, stock_symbol):
    stock_data = data.loc[data['StockSymbol'] == stock_symbol].copy()
    
    # Calculate the moving average
    stock_data['SMA'] = calculate_sma(stock_data)
    
    # Calculate the percentage change
    stock_data['PercentageChange'] = stock_data['ClosePrice'].pct_change() * 100
    
    # Calculate average percentage change over the last 5 days
    recent_changes = stock_data['PercentageChange'].tail(5)
    average_change = recent_changes.mean()

    # Get current price and SMA
    current_price = stock_data['ClosePrice'].iloc[-1]
    sma_value = stock_data['SMA'].iloc[-1]
    
    # Trend logic based on percentage change and SMA comparison
    trend_info = {
        'direction': 'upward' if average_change > 0 else 'downward' if average_change < 0 else 'stable',
        'price_vs_sma': 'above' if current_price > sma_value else 'below',
        'average_change': average_change,
        'volatility': stock_data['PercentageChange'].std(),
        'current_price': current_price,
        'sma_value': sma_value,
        'recent_changes': recent_changes.tolist()
    }
    
    return trend_info, stock_data

# Step 6: Generate dynamic responses based on stock trends and news sentiment
def generate_response(trend_info, sentiment):
    responses = {
        "upward": [
            "The stock is on an upward trend with strong momentum!",
            "Great news! The stock price is climbing steadily."
        ],
        "downward": [
            "Caution! The stock is on a downward trend.",
            "It looks like the stock is losing value. Consider your options."
        ],
        "stable": [
            "The stock is stable, showing no significant price movements.",
            "The stock is currently stable with little change."
        ]
    }
    
    sentiment_feedback = {
        "positive": "Recent news sentiment is positive, which could indicate potential growth!",
        "negative": "Recent news sentiment is negative, which might suggest further declines.",
        "neutral": "The news sentiment is neutral, indicating mixed signals."
    }
    
    # Select a random response based on trend and append sentiment feedback
    response = random.choice(responses[trend_info['direction']])
    response += f" Current price: ${trend_info['current_price']:.2f}, SMA: ${trend_info['sma_value']:.2f}. The stock is currently trading {trend_info['price_vs_sma']} its moving average."
    
    if trend_info['volatility'] > 2:
        response += " Be cautious, as the stock is quite volatile!"
    
    response += f" {sentiment_feedback[sentiment]}"
    
    return response

# Step 7: Chatbot interface
def chatbot_interface(data):
    print("Welcome to the AI-powered Stock Advisor Chatbot!")
    print("Ask me about stock trends and recent news.\n")
    
    while True:
        user_input = input("You: ").lower().strip()
        
        if user_input in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        if "stock" in user_input:
            stock_symbol = input("Please enter the stock symbol (e.g., AAPL): ").upper()
            
            if stock_symbol in data['StockSymbol'].unique():
                # Analyze stock trends
                trend_info, stock_data = analyze_stock_trend(data, stock_symbol)
                
                # Fetch and analyze news articles
                headlines = fetch_news(stock_symbol)
                sentiment = analyze_sentiment(headlines)
                
                # Generate response
                response = generate_response(trend_info, sentiment)
                print(f"Chatbot: {response}")
                
                # Display the stock chart
                plt.figure(figsize=(10, 6))
                plt.plot(stock_data['Date'], stock_data['ClosePrice'], label='Close Price')
                plt.plot(stock_data['Date'], stock_data['SMA'], label='7-day SMA', color='orange')
                plt.title(f"Stock Price and SMA for {stock_symbol}")
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid()
                plt.show()
            else:
                print(f"Chatbot: Sorry, I don't have data for the stock symbol '{stock_symbol}'.")
        else:
            print("Chatbot: I can help with stock trends and recent news. Ask about a stock!")

# Step 8: Main function to run the chatbot
if __name__ == "__main__":
    stock_data = create_dataset()
    chatbot_interface(stock_data)
