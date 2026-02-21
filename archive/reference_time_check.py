import z_util as zu
df = zu.load_parquet('returns')
print(df['time'].head(20))
print(f"Min time: {df['time'].min()}, Max time: {df['time'].max()}")