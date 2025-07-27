import pandas as pd
import sqlalchemy, pymysql, psycopg2
from sqlalchemy import create_engine
import mysql.connector

azure_engine = create_engine(
    "mysql+mysqlconnector://zishaan:VS9eo23cKASvhQ@sra.mysql.database.azure.com:3306/source"
)
pg_engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://optionsDB:z1sh0PT10Neleph%40ntSQL@34.150.156.184:5432/optionsDB"
)

with azure_engine.connect() as conn:
    tables_df = pd.read_sql("SHOW TABLES;", conn)

table_list = ['av_minute', 'av_minute_depr', 'iqfeed_minute', 'quantquote_minute']
chunk_size = 100
print("Beginning to load tables to PostgreSQL")
for i in range(len(table_list)):
    print("1")
    for chunk_number, chunk in enumerate(pd.read_sql(f"SELECT * FROM {table_list[i]}", azure_engine, chunksize=chunk_size), start=1):
        print("Beginning next chunk")
        chunk.to_sql(
            name=table_list[i],
            con=pg_engine,
            if_exists='replace',
            index=False,
            method='multi'
        )
        print(f"Table {table_list[i]} chunk {chunk_number} has been loaded to PostgreSQL")
    print(f"Table {table_list[i]} has been loaded to PostgreSQL")
print("All tables have been loaded to PostgreSQL")