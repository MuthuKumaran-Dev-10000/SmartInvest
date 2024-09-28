import yfinance as yf
import pandas as pd

# Define the stocks or assets to analyze
assets = [
    # US Stocks
    'AAPL', 'MSFT', 'TSLA', 'GOOG', 'SPY', 'AMZN', 'FB', 'NFLX', 'NVDA', 'AMD',
    'INTC', 'PYPL', 'SQ', 'DIS', 'V', 'MA', 'JPM', 'BAC', 'GS', 'C', 'WFC', 'XOM', 
    'CVX', 'PFE', 'JNJ', 'MRNA', 'UNH', 'PEP', 'KO', 'COST', 'WMT', 'HD', 'LOW', 
    'TGT', 'NKE', 'SBUX', 'MCD', 'MMM', 'BA', 'CAT', 'GE', 'LMT', 'RTX', 'IBM',
    
    # International Stocks
    'BABA', 'TCEHY', 'NIO', 'JD', 'PDD', 'SHOP', 'VWO', 'BIDU', 'NTES', 'TSM', 'SNP',
    'BBL', 'VALE', 'RIO', 'SCCO', 'PBR', 'EC', 'YPF', 'TM', 'HMC', 'SONY', 'NTDOY',
    
    # ETFs
    'VOO', 'IVV', 'VTI', 'VUG', 'ARKK', 'XLF', 'XLE', 'XLI', 'XLV', 'XLY', 'XLK',
    'DIA', 'QQQ', 'IWM', 'EEM', 'GLD', 'SLV', 'GDX', 'VWO', 'EFA', 'SPHD', 'SPLV',
    
    # Bonds & Fixed Income
    'TLT', 'IEF', 'SHY', 'LQD', 'BND', 'AGG', 'HYG', 'TIP', 'MBB', 'BSV', 'VGSH', 'VMBS',
    
    # Commodities
    'USO', 'UNG', 'GLD', 'SLV', 'PALL', 'PLAT', 'CORN', 'SOYB', 'WEAT', 'DBA', 'NIB',
    
    # REITs (Real Estate)
    'VNQ', 'SCHH', 'IYR', 'SPG', 'PLD', 'PSA', 'DLR', 'AMT', 'EQIX', 'O', 'FRT', 'VTR',
    
    # Technology
    'ADBE', 'CRM', 'NOW', 'ORCL', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'TWTR', 'SNAP',
    
    # Healthcare
    'ABBV', 'ABT', 'BMY', 'GILD', 'LLY', 'ZTS', 'ISRG', 'DHR', 'MDT', 'SYK', 'CVS',
    
    # Consumer Goods & Services
    'PG', 'CL', 'KMB', 'UL', 'HENKY', 'TGT', 'HD', 'LOW', 'ROST', 'TJX', 'DG', 'DLTR',
    
    # Financials
    'BLK', 'SCHW', 'TROW', 'MS', 'BAC', 'C', 'PNC', 'USB', 'BK', 'AXP', 'V', 'MA', 'PYPL',
    
    # Energy
    'COP', 'EOG', 'SLB', 'HAL', 'KMI', 'OKE', 'WMB', 'PSX', 'VLO', 'MPC',
    
    # Industrials
    'DE', 'UNP', 'NSC', 'FDX', 'UPS', 'CSX', 'DAL', 'LUV', 'UAL', 'AAL',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'ES', 'XEL', 'SRE', 'AWK'
]
# You can add more assets here

# Define a function to fetch data with error handling
def fetch_asset_data(symbol):
    print(f"Fetching data for {symbol}...")
    try:
        data = yf.download(symbol, period='1y', interval='1d')
        if not data.empty:
            data['Symbol'] = symbol
        else:
            print(f"No data for {symbol}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()  # Return empty dataframe on failure
    return data

# Function to calculate buy/sell signals based on a simple optimization technique
def find_buy_sell_signals(data, window=20):
    data['Moving_Avg'] = data['Close'].rolling(window=window).mean()

    # Find Buy signals (Min Dataset)
    data['Buy_Signal'] = data['Close'] < data['Moving_Avg']
    
    # Find Sell signals (Max Dataset)
    data['Sell_Signal'] = data['Close'] > data['Moving_Avg']

    return data

# Fetch data for all assets and concatenate
all_data = pd.DataFrame()
for asset in assets:
    df = fetch_asset_data(asset)
    all_data = pd.concat([all_data, df])

# Reset index
all_data.reset_index(inplace=True)

# Apply the buy/sell signal strategy for each asset
all_data = find_buy_sell_signals(all_data)

# Create Min Dataset (Buying Opportunities)
min_dataset = all_data[all_data['Buy_Signal']].copy()
min_dataset.drop(['Buy_Signal', 'Sell_Signal', 'Moving_Avg'], axis=1, inplace=True)

# Create Max Dataset (Selling Opportunities)
max_dataset = all_data[all_data['Sell_Signal']].copy()
max_dataset.drop(['Buy_Signal', 'Sell_Signal', 'Moving_Avg'], axis=1, inplace=True)

# Ensure both datasets have a max of 100,000 rows
min_dataset = min_dataset.head(100000)
max_dataset = max_dataset.head(100000)

# Save the datasets to Excel files
min_dataset.to_excel('Min_Dataset.xlsx', index=False)
max_dataset.to_excel('Max_Dataset.xlsx', index=False)

print("Min_Dataset.xlsx and Max_Dataset.xlsx have been saved successfully.")
