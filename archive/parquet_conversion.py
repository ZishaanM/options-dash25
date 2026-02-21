import pandas as pd
import z_util as zu
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os
import gc

# =============================================================================
# MANUAL RESUME CONTROL
# If the script crashes and you want to resume from a specific point:
#   1. Set START_FROM_TABLE to the table name (e.g., "av_minute")
#   2. Set START_FROM_OFFSET to the row number to resume from
#   3. Tables before START_FROM_TABLE will be SKIPPED entirely
#
# To start fresh: set both to None and delete all .parquet files manually
# =============================================================================
START_FROM_TABLE = None  # e.g., "av_minute" or None to process all
START_FROM_OFFSET = None  # e.g., 21600000 or None to auto-detect from existing file

# =============================================================================
CHUNK_SIZE = 100000  # 100k rows per chunk
tables = ["av_minute", "iqfeed_minute", "quantquote_minute", "av_minute_depr", "returns"]

# Connect once
print("Connecting to GCP...")
engine = zu.connect_gcp()['engine']

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting migration...")
print(f"Chunk size: {CHUNK_SIZE:,} rows\n")

# Skip tables if resuming from a specific table
skip_until_found = START_FROM_TABLE is not None
tables_to_process = []
for t in tables:
    if skip_until_found:
        if t == START_FROM_TABLE:
            skip_until_found = False
            tables_to_process.append(t)
        else:
            print(f"Skipping {t} (resuming from {START_FROM_TABLE})")
    else:
        tables_to_process.append(t)

for table_name in tables_to_process:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Starting {table_name} ===")
    
    # Get total row count for ETA
    print(f"  Querying total row count...")
    total_in_table = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", engine).iloc[0]['cnt']
    print(f"  Total rows in {table_name}: {total_in_table:,}")
    
    # Determine starting offset
    filepath = f"{table_name}.parquet"
    if START_FROM_TABLE == table_name and START_FROM_OFFSET is not None:
        # Manual override
        offset = START_FROM_OFFSET
        print(f"  MANUAL RESUME: Starting from row {offset:,}")
        # Truncate existing file to specified offset
        if os.path.exists(filepath):
            print(f"  Truncating existing file to {offset:,} rows...")
            existing_df = pd.read_parquet(filepath)
            existing_df = existing_df.iloc[:offset]
            existing_df.to_parquet(filepath + ".tmp", engine='pyarrow', compression='snappy')
            os.remove(filepath)
            os.rename(filepath + ".tmp", filepath)
            del existing_df
            gc.collect()
    elif os.path.exists(filepath):
        # Auto-detect from existing file
        existing_df = pd.read_parquet(filepath)
        offset = len(existing_df)
        print(f"  Found existing file with {offset:,} rows. Resuming from there.")
        del existing_df
        gc.collect()
    else:
        offset = 0
    
    if offset >= total_in_table:
        print(f"  Already complete! Skipping.")
        continue
    
    total_rows = offset
    start_time = datetime.now()
    
    # Open existing file in append mode or create new
    writer = None
    if offset > 0:
        # Read existing file to get schema, then append
        existing_table = pq.read_table(filepath)
        schema = existing_table.schema
        # We'll write to a temp file and combine at the end
        writer = pq.ParquetWriter(filepath + ".new", schema, compression='snappy')
        writer.write_table(existing_table)
        del existing_table
        gc.collect()
    
    while True:
        pct = (offset / total_in_table) * 100
        elapsed = (datetime.now() - start_time).total_seconds()
        rows_done_this_session = offset - (total_rows - (offset - total_rows)) if total_rows > 0 else 0
        
        # ETA calculation
        if offset > total_rows and elapsed > 0:
            rows_per_sec = (offset - total_rows + CHUNK_SIZE) / elapsed
            remaining_rows = total_in_table - offset
            eta_seconds = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0
            eta_str = f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Fetching {offset:,} to {offset + CHUNK_SIZE:,}... ({pct:.1f}% | {eta_str})")
        
        # Force server-side pagination with LIMIT/OFFSET
        query = f"SELECT * FROM {table_name} LIMIT {CHUNK_SIZE} OFFSET {offset}"
        chunk = pd.read_sql(query, engine)
        
        rows_in_chunk = len(chunk)
        
        if rows_in_chunk == 0:
            break  # No more data
        
        # Convert to Arrow table
        arrow_table = pa.Table.from_pandas(chunk)
        
        # Initialize writer on first chunk (if starting fresh)
        if writer is None:
            writer = pq.ParquetWriter(filepath + ".new" if offset > 0 else filepath, arrow_table.schema, compression='snappy')
        
        writer.write_table(arrow_table)
        
        offset += rows_in_chunk
        total_rows = offset
        
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Wrote {rows_in_chunk:,} rows (Total: {total_rows:,} / {total_in_table:,})")
        
        # Free memory explicitly
        del chunk
        del arrow_table
        gc.collect()
        
        # If we got fewer rows than chunk size, we're done
        if rows_in_chunk < CHUNK_SIZE:
            break
    
    if writer:
        writer.close()
        # If we were appending, replace original with new file
        if os.path.exists(filepath + ".new"):
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(filepath + ".new", filepath)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {table_name}: {total_rows:,} total rows\n")

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === All tables migrated ===\n")

# Verification
print("Verifying files...")
for table_name in tables:
    filepath = f"{table_name}.parquet"
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        print(f"\n{table_name}.parquet: {len(df):,} rows")
        print(df.head(3))
        del df
        gc.collect()
    else:
        print(f"\n{table_name}.parquet: NOT FOUND")
