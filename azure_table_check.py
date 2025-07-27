import pandas as pd
import sqlalchemy, pymysql, psycopg2
from sqlalchemy import create_engine

from sqlalchemy import create_engine, text

import pandas as pd
from sqlalchemy import create_engine

azure_engine = create_engine(
    "mysql+pymysql://zishaan:VS9eo23cKASvhQ@sra.mysql.database.azure.com:3306/source"
)

table = 'quantquote_minute'
timestamp_col = 'date'
'''
result = pd.read_sql(
    f"SELECT * FROM {table} LIMIT 15",
    azure_engine
)
print(f"Columns: {result}")

result = pd.read_sql(
    f"SELECT MIN({timestamp_col}) as min_ts FROM {table}",
    azure_engine
)
print(f"First timestamp in {table}: {result['min_ts'][0]}")

'''
result = pd.read_sql(
    f"SELECT MAX({timestamp_col}) as max_ts FROM {table}",
    azure_engine
)
print(f"Last timestamp in {table}: {result['max_ts'][0]}")


# result = pd.read_sql(
#     f"SELECT COUNT(*) as total_rows FROM {table}",
#     azure_engine
# )
# print(f"Row count: {result['total_rows'][0]}")
