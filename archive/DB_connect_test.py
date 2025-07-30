# This is a test to see if I can connect to the database
from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://optionsDB:z1sh0PT10Neleph%40ntSQL@34.150.156.184:5432/optionsDB")

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1;"))
    print(result.fetchone())
