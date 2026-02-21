import z_util as zu
import pandas as pd
import pyarrow.parquet as pq
import os

engine = zu.connect_gcp()['engine']

tables = ["av_minute", "returns"]
for table in tables:
    # GCP count
    gcp_count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", engine).iloc[0]['cnt']
    print(f"{table}:")
    print(f"  GCP:     {gcp_count:,} rows")
    
    # Parquet count (metadata only - no memory needed!)
    filepath = f"{table}.parquet"
    if os.path.exists(filepath):
        parquet_count = pq.read_metadata(filepath).num_rows
        match = "✓" if gcp_count == parquet_count else "✗ MISMATCH"
        print(f"  Parquet: {parquet_count:,} rows {match}")
    else:
        print(f"  Parquet: NOT FOUND")