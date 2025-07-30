import pandas as pd
import sqlalchemy, pymysql, psycopg2
from sqlalchemy import create_engine

azure_engine = create_engine(
    "mysql+pymysql://zishaan:VS9eo23cKASvhQ@sra.mysql.database.azure.com:3306/source"
)
pg_engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://optionsDB:z1sh0PT10Neleph%40ntSQL@34.150.156.184:5432/optionsDB"
)

table_list = ['av_minute', 'av_minute_depr', 'iqfeed_minute', 'quantquote_minute']
table = 'quantquote_minute'
timestamp_col = 'date'  # Change if your column is named differently, use SELECT statement to do 4 in one?

print("Beginning to load tables to PostgreSQL by day")
#for table in table_list:
    # Get min and max timestamp
min_max = pd.read_sql(
    f"SELECT MIN({timestamp_col}) as min_ts, MAX({timestamp_col}) as max_ts FROM {table}",
    azure_engine
)
min_ts = min_max['min_ts'][0]
max_ts = min_max['max_ts'][0]
#start_date = pd.Timestamp('2021-07-09')  # Desired start date

'''
first_100 = pd.read_sql(
    f"SELECT * FROM {table} LIMIT 100",
    azure_engine
)
if not first_100.empty:
    first_100.to_sql(
        name=table,
        con=pg_engine,
        if_exists='append',
        index=False,
        method='multi'
    )
    print(f"Uploaded first 100 rows of {table} to GCP PostgreSQL.")
else:
    print(f"No data found in {table} to upload.")

'''
if pd.isnull(min_ts) or pd.isnull(max_ts):
    print(f"Table {table} is empty, skipping.")
else:
    current_start = int(min_ts)
    current_end = int(max_ts)
    while current_start <= current_end:
        day_str = str(current_start)
        query = (
            f"SELECT * FROM {table} "
            f"WHERE {timestamp_col} = '{day_str}'"
        )
        chunk = pd.read_sql(query, azure_engine)
        if not chunk.empty:
            chunk.to_sql(
                name=table,
                con=pg_engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            print(f"Table {table}: loaded data for {day_str}")
        # Increment to next day
        next_day = pd.to_datetime(str(current_start), format='%Y%m%d') + pd.Timedelta(days=1)
        current_start = int(next_day.strftime('%Y%m%d'))
    print(f"Table {table} has been loaded to PostgreSQL by day")
#'''