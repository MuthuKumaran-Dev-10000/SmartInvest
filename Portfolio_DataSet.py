import pandas as pd
import numpy as np
import random

# Constants for the logic
ASSET_TYPES = ['Stocks', 'Bonds', 'ETFs', 'Real Estate', 'Cash']
RISK_LEVELS = ['Low', 'Moderate', 'High']
SECTORS = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods', 'Real Estate']

# Function to generate a random asset allocation
def generate_asset_allocation():
    allocations = np.random.dirichlet(np.ones(len(ASSET_TYPES)), size=1)[0] * 100
    return dict(zip(ASSET_TYPES, allocations))

# Function to assign a risk level based on asset allocation
def calculate_risk_level(allocation):
    stock_weight = allocation['Stocks']
    bond_weight = allocation['Bonds']
    
    if stock_weight > 50:
        return 'High'
    elif bond_weight > 40:
        return 'Low'
    else:
        return 'Moderate'

# Function to simulate performance based on risk and asset allocation
def simulate_performance(risk_level):
    if risk_level == 'High':
        return round(np.random.normal(loc=10, scale=5), 2)  # Higher risk, higher returns (avg 10%, SD 5%)
    elif risk_level == 'Moderate':
        return round(np.random.normal(loc=5, scale=3), 2)   # Moderate risk, moderate returns (avg 5%, SD 3%)
    else:
        return round(np.random.normal(loc=3, scale=2), 2)   # Low risk, lower returns (avg 3%, SD 2%)

# Function to assign sector exposure
def generate_sector_exposure():
    allocations = np.random.dirichlet(np.ones(len(SECTORS)), size=1)[0] * 100
    return dict(zip(SECTORS, allocations))

# Function to generate a random time horizon
def generate_time_horizon():
    return random.choice([5, 10, 15, 20, 25, 30])

# Generate dataset for 1000 portfolios
portfolios = []
for i in range(1000):
    allocation = generate_asset_allocation()
    risk_level = calculate_risk_level(allocation)
    performance = simulate_performance(risk_level)
    sector_exposure = generate_sector_exposure()
    time_horizon = generate_time_horizon()
    
    # Portfolio data
    portfolio = {
        'Portfolio_ID': f'PF_{i+1}',
        'Total_Value': round(random.uniform(50_000, 500_000), 2),  # Portfolio value between 50k and 500k
        'Risk_Level': risk_level,
        'Stocks_Allocation': allocation['Stocks'],
        'Bonds_Allocation': allocation['Bonds'],
        'ETFs_Allocation': allocation['ETFs'],
        'Real_Estate_Allocation': allocation['Real Estate'],
        'Cash_Allocation': allocation['Cash'],
        'Performance_%': performance,
        'Time_Horizon_Years': time_horizon,
        'Technology_Exposure': sector_exposure['Technology'],
        'Healthcare_Exposure': sector_exposure['Healthcare'],
        'Finance_Exposure': sector_exposure['Finance'],
        'Energy_Exposure': sector_exposure['Energy'],
        'Consumer_Goods_Exposure': sector_exposure['Consumer Goods'],
        'Real_Estate_Exposure': sector_exposure['Real Estate']
    }
    portfolios.append(portfolio)

# Convert list of portfolios to DataFrame
df = pd.DataFrame(portfolios)

# Export the data to an Excel file
df.to_excel('portfolios_dataset.xlsx', index=False)

print("Dataset generated and saved as 'portfolios_dataset.xlsx'")
