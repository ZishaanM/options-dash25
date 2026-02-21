"""
FAST Parquet Conversion using PostgreSQL COPY command
This is 5-10x faster than LIMIT/OFFSET queries.

Two-step process:
1. Stream data directly from PostgreSQL using COPY TO STDOUT (very fast)
2. Convert to Parquet locally

No need for psql installed - uses psycopg2's copy_expert.
"""

import pandas as pd
import z_util as zu
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os
import io
import gc

# =============================================================================
# CONFIGURATION
# =============================================================================
tables = ["av_minute", "iqfeed_minute", "quantquote_minute", "av_minute_depr", "returns"]

# Skip tables you've already completed (set to table name to start from)
START_FROM_TABLE = None  # e.g., "iqfeed_minute" to skip av_minute

# =============================================================================

print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to GCP...")

# Get raw psycopg2 connection (needed for copy_expert)
con_info = zu.connect_gcp()
engine = con_info['engine']

# Determine which tables to process
if START_FROM_TABLE:
    start_idx = tables.index(START_FROM_TABLE)
    tables_to_process = tables[start_idx:]
    print(f"Skipping tables before {START_FROM_TABLE}")
else:
    tables_to_process = tables

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting FAST migration...")
print(f"Tables to process: {tables_to_process}\n")

for table_name in tables_to_process:
    filepath = f"{table_name}.parquet"
    csv_filepath = f"{table_name}.csv"
    
    # Skip if parquet already exists (use metadata to avoid loading entire file)
    if os.path.exists(filepath):
        parquet_meta = pq.read_metadata(filepath)
        existing_rows = parquet_meta.num_rows
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {table_name}.parquet already exists ({existing_rows:,} rows). Skipping.")
        print(f"  (Delete the file manually if you want to re-download)\n")
        continue
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === Starting {table_name} ===")
    
    # Get row count for progress
    total_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", engine).iloc[0]['cnt']
    print(f"  Total rows: {total_rows:,}")
    
    start_time = datetime.now()
    
    # Use COPY command for fast streaming export
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Streaming from PostgreSQL (this is the fast part)...")
    
    # Get a raw connection for copy_expert
    raw_conn = engine.raw_connection()
    cursor = raw_conn.cursor()
    
    # Stream directly to CSV file (fastest method)
    # Use SELECT syntax to support both tables AND views
    copy_sql = f"COPY (SELECT * FROM {table_name}) TO STDOUT WITH CSV HEADER"
    
    with open(csv_filepath, 'w', encoding='utf-8', newline='') as f:
        cursor.copy_expert(copy_sql, f)
    
    cursor.close()
    raw_conn.close()
    
    csv_time = datetime.now()
    csv_elapsed = (csv_time - start_time).total_seconds()
    csv_size_mb = os.path.getsize(csv_filepath) / (1024 * 1024)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] CSV export complete: {csv_size_mb:.1f} MB in {csv_elapsed:.0f}s ({csv_size_mb/csv_elapsed:.1f} MB/s)")
    
    # Convert CSV to Parquet in chunks (memory efficient)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Converting CSV to Parquet...")
    
    chunk_size = 500000  # Larger chunks OK for local processing
    writer = None
    rows_written = 0
    
    for chunk in pd.read_csv(csv_filepath, chunksize=chunk_size, low_memory=False):
        arrow_table = pa.Table.from_pandas(chunk)
        
        if writer is None:
            writer = pq.ParquetWriter(filepath, arrow_table.schema, compression='snappy')
        
        writer.write_table(arrow_table)
        rows_written += len(chunk)
        
        pct = (rows_written / total_rows) * 100
        print(f"    Converted {rows_written:,} / {total_rows:,} rows ({pct:.0f}%)")
        
        del chunk
        del arrow_table
        gc.collect()
    
    if writer:
        writer.close()
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    parquet_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Parquet complete: {parquet_size_mb:.1f} MB")
    print(f"  Compression ratio: {csv_size_mb/parquet_size_mb:.1f}x smaller than CSV")
    print(f"  Total time: {total_elapsed:.0f}s ({total_rows/total_elapsed:.0f} rows/sec)")
    
    # Delete CSV to save disk space
    os.remove(csv_filepath)
    print(f"  Deleted temporary CSV file")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {table_name}\n")
    gc.collect()

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === All tables migrated ===\n")

# Verification
print("Verifying files...")
for table_name in tables:
    filepath = f"{table_name}.parquet"
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\n{table_name}.parquet: {len(df):,} rows, {size_mb:.1f} MB")
        print(df.head(3))
        del df
        gc.collect()
    else:
        print(f"\n{table_name}.parquet: NOT FOUND")
