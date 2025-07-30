import sqlalchemy as sa
import pandas as pd

import z_util as zu

table_name = "calc_returns"
columns_str = "date DATE, ret. from open DOUBLE PRECISION, ret. from prev. close DOUBLE PRECISION, ret. from high DOUBLE PRECISION, ret. from low DOUBLE PRECISION, ret. to today's close DOUBLE PRECISION"
query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});"
pd.read_sql(query)


# Connect to the database
con = zu.connect_gcp()
engine = con['engine']

# Get unique dates from quantquote_minute table
dates_df = pd.read_sql("SELECT DISTINCT date FROM quantquote_minute ORDER BY date ASC", engine)

# Insert dates into calc_returns table if not already present
for date in dates_df['date']:
    insert_query = f"""
        INSERT INTO {table_name} (date)
        VALUES ('{date}')
        ON CONFLICT (date) DO NOTHING;
    """
    with engine.begin() as conn:
        conn.execute(sa.text(insert_query))





