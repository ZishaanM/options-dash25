import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

# Get current date and time in EST
est = pytz.timezone('US/Eastern')
current_time = datetime.now(est)
current_date = current_time.strftime('%Y-%m-%d')
print(current_date)
ticker = "TSLA"
# Get intraday data with specific times
v = yf.download(ticker, 
                start=current_date,  # Today's date
                end=current_time,    # Today's date
                interval="1m")

# Convert index to EST and format
v.index = v.index.tz_convert(est)
v.index = v.index.strftime('%Y-%m-%d %H:%M:%S EST')

# Save to CSV
v.to_csv('stock_data.csv')
print("\nData has been saved to 'stock_data.csv'")
print(f"Data from {current_date} up to current time ({current_time.strftime('%H:%M:%S')} EST)")

# Create time series graph
plt.figure(figsize=(12, 6))
plt.plot(v.index, v['Close'], label='Closing Price')
plt.title(f'{ticker} Stock Price ({current_date})')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




