import pandas as pd

# Load the dataset
data_set = pd.read_csv('/home/khudi/Desktop/autogen_ds/datasets/data_set.csv')

# Perform forecasting analysis using the fetched data
# This is where you can use your forecasting models and techniques

# Example of Forecasting Model
# For example, using a simple moving average for forecasting sales for a specific SKU
sku_id = 10046
sales_data_sku = data_set[data_set['SKU ID'] == sku_id]
sales_data_sku['Date'] = pd.to_datetime(sales_data_sku['Date'])  # Convert Date to datetime format
sales_data_sku = sales_data_sku.set_index('Date')

# Calculate the moving average for sales
sales_data_sku['MA_7'] = sales_data_sku['Sales'].rolling(window=7).mean()
forecasted_sales = sales_data_sku[['Sales', 'MA_7']].tail(10)  # Forecasted sales for the last 10 days

print(forecasted_sales)

# You can use more sophisticated forecasting techniques such as ARIMA, Prophet, etc. based on your analysis requirement
