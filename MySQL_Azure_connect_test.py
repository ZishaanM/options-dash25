from sqlalchemy import create_engine
import pandas as pd

pg_engine = create_engine(
    "postgresql+psycopg2://optionsDB:z1sh0PT10Neleph%40ntSQL@34.150.156.184:5432/optionsDB"
)

with pg_engine.connect() as conn:
    df = pd.read_sql("SHOW TABLES;", conn)
    print(df)
