from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy.stats import zscore

pg_engine = create_engine(
    "postgresql+psycopg2://optionsDB:z1sh0PT10Neleph%40ntSQL@34.150.156.184:5432/optionsDB"
)
table = "quantquote_minute"


'''
#Volume throughout day
df = pd.read_sql(f"SELECT date, volume, time FROM {table} ORDER BY date ASC", pg_engine)
df['date'] = pd.to_datetime(df['date'])
df['time_str'] = df['time'].astype(str).str.zfill(4)
df['time_fmt'] = pd.to_datetime(df['time_str'], format='%H%M').dt.time
df['hour'] = df['time_str'].str[:2]
volume_by_hour = df.groupby('hour')['volume'].mean()
plt.figure(figsize=(10,5))
plt.plot(volume_by_hour.index, volume_by_hour.values)
plt.title('Average Intraday Volume Profile')
plt.xlabel('Hour (24h)')
plt.ylabel('Average Volume')
plt.show()



# Histogram of Returns
df = pd.read_sql(f"SELECT date, close FROM {table} WHERE time = '1600' ORDER BY date ASC", pg_engine)
df['returns'] = df['close'].pct_change()
print(df['returns'].head())
plt.hist(df['returns'], bins=50, edgecolor='black')
plt.title(f'Histogram of Daily Returns ({table})')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()
'''

#Time Series of Vol over the year
df = pd.read_sql(
    f"SELECT date, close, volume FROM {table} "
    "WHERE time = '1600' ORDER BY date", pg_engine
)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
window = 20
df['vol'] = df['log_ret'].rolling(window).std()
df['vol_ann'] = df['vol'] * np.sqrt(252)
df['month_day'] = df['date'].dt.strftime('%m-%d')
avg_vol_by_day = df.groupby('month_day')['vol_ann'].mean()
volume_avg = df.groupby('month_day')['volume'].mean()
month_days = [datetime.datetime.strptime(f'2000-{md}', '%Y-%m-%d') for md in avg_vol_by_day.index]
plt.figure(figsize=(14,6))
scaled_vol = volume_avg.values / 1e7
vol_z = zscore(avg_vol_by_day.values)
volu_z = zscore(volume_avg.values)
pearson_corr = df.groupby('month_day')[['vol_ann', 'volume']].mean().corr().iloc[0,1]
spearman_corr = df.groupby('month_day')[['vol_ann', 'volume']].mean().corr(method='spearman').iloc[0,1]
print(f'Pearson correlation coefficient {pearson_corr.round(6)}\nSpearman correlation coefficient {spearman_corr.round(6)}')
plt.plot(month_days, vol_z, label='Volatility (Z-score)', color='blue')
plt.plot(month_days, volu_z, label='Volume (Z-score)', color='red')
plt.title(f'Average Annualized Volatility and Volume by Calendar Day ({table})')
plt.xlabel('Calendar Day (Jan 1 to Dec 31)')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
plt.show()