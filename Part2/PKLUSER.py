# Load the functions from the pickle file

import dill
import yfinance as yf  # Import yfinance
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import Sentiment Analyzer

# Load functions from the pickle file
with open('stock_functions.pkl', 'rb') as f:
    functions = dill.load(f)

# Extract functions
fetch_stock_data = functions["fetch_stock_data"]
analyze_sentiment = functions["analyze_sentiment"]
generate_response = functions["generate_response"]

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
            try:
                stock_data = fetch_stock_data(stock_symbol)

                if not stock_data.empty:
                    # Mocked headlines for sentiment analysis
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
            except Exception as e:
                print(f"Chatbot: An error occurred: {e}")
        else:
            print("Chatbot: I can help with stock trends and recent news. Ask about a stock!")

# Main function to run the chatbot
if __name__ == "__main__":
    chatbot_interface()
