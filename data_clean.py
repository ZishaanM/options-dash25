from sqlalchemy import create_engine
import pandas as pd

pg_engine = create_engine(
    "postgresql+psycopg2://optionsDB:z1sh0PT10Neleph%40ntSQL@34.150.156.184:5432/optionsDB"
)

# Specify your table name and the relevant columns
table_name = 'quantquote_minute'  # Change as needed
stock_price_col = 'close'         # Change to your stock price column name

with pg_engine.connect() as conn:
    # Get column names
    cols = pd.read_sql(
        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'", conn
    )
    col_list = cols['column_name'].tolist()
    # Build condition for any NULL
    null_condition = " OR ".join([f"{col} IS NULL" for col in col_list])
    # NaN and impossible value conditions for stock_price_col
    nan_condition = f"{stock_price_col}::text = 'NaN'"
    impossible_condition = f"{stock_price_col} < 0"
    # Combine all conditions
    all_conditions = []
    if null_condition:
        all_conditions.append(f"({null_condition})")
    all_conditions.append(f"({nan_condition})")
    all_conditions.append(f"({impossible_condition})")
    where_clause = " OR ".join(all_conditions)
    query = f"SELECT COUNT(*) as bad_row_count FROM {table_name} WHERE {where_clause}"
    result = pd.read_sql(query, conn)
    print(f"Number of rows with null, NaN, or impossible values: {result['bad_row_count'][0]}")



'''
# 1. Delete rows with NULLs in any column
with pg_engine.connect() as conn:
    # Get column names
    cols = pd.read_sql(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'", conn)
    col_list = cols['column_name'].tolist()
    # Build condition for any NULL
    null_condition = " OR ".join([f"{col} IS NULL" for col in col_list])
    if null_condition:
        conn.execute(f"DELETE FROM {table_name} WHERE {null_condition}")
    conn.execute(f"DELETE FROM {table_name} WHERE {stock_price_col}::text = 'NaN'")
    conn.execute(f"DELETE FROM {table_name} WHERE {stock_price_col} < 0")
'''