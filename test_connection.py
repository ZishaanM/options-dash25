from dotenv import load_dotenv
import psycopg2
import os
from pathlib import Path

#load_dotenv()  # Load variables from .env
load_dotenv(dotenv_path=".env")

# Fetch variables
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")

print(f"DB_HOST value: {host}")
print(f"DB_USER value: {user}")

# Try connecting
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    print("Connection successful.")
    conn.close()
except Exception as e:
    print("Connection failed:")
    print(e)


