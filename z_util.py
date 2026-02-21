import os
import logging
import logging.handlers
from functools import lru_cache

import pandas as pd
import numpy as np
import sqlalchemy as sa

# =============================================================================
# PARQUET DATA LOADING (Local storage - replaces GCP)
# =============================================================================

# Cache for loaded parquet dataframes
_parquet_cache = {}

def load_parquet(table_name: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Load a parquet file by table name with caching.
    
    Args:
        table_name: Name of the table (e.g., 'returns', 'av_minute')
        force_reload: If True, reload from disk even if cached
    
    Returns:
        pd.DataFrame: The loaded data
    """
    global _parquet_cache
    
    if table_name in _parquet_cache and not force_reload:
        return _parquet_cache[table_name]
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, f"{table_name}.parquet")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    _parquet_cache[table_name] = df
    
    return df

def clear_parquet_cache(table_name: str = None):
    """
    Clear the parquet cache.
    
    Args:
        table_name: If provided, only clear that table. Otherwise clear all.
    """
    global _parquet_cache
    if table_name:
        _parquet_cache.pop(table_name, None)
    else:
        _parquet_cache.clear()

# =============================================================================


def init_logging(logger, log_file='main.log', central_error_log='cron.log'):
    """
    Initialize logging configuration with both script-specific and centralized error logging.
   
    Args:
        logger: Logger instance to configure
        log_file: Path to script-specific log file
    """
    class OCSPFilter(logging.Filter):
        def filter(self, record):
            # Filter out messages from the OCSP logger
            return not record.name == 'snowflake.connector.ocsp_snowflake'
       
    logFormatter = logging.Formatter(
        fmt='%(asctime)s %(filename)-20.20s->%(funcName)-30.30s:%(lineno)-4s [%(levelname)-8.8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
   
    if not logger.handlers:
        logger.setLevel(logging.INFO)
       
        # Console handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
       
        # Script-specific file handler
        fileHandler = logging.handlers.RotatingFileHandler(
            os.path.join('log',log_file),
            maxBytes=1 * 1024 * 1024,
            backupCount=0
            )
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
       
        # Central error handler
        error_log_path = central_error_log
        error_handler = logging.FileHandler(error_log_path)
        error_handler.setFormatter(logFormatter)
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(OCSPFilter())
        logger.addHandler(error_handler)
   
    logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
    logging.getLogger("snowflake.connector").setLevel(logging.ERROR)
    logging.getLogger("azure").setLevel(logging.ERROR)

def get_logger(name):
    """
    Get a configured logger instance.
   
    Args:
        name: Name for the logger (typically __name__)
        log_file: Optional log file path. If None, uses name.log
   
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger() #get the root logger
    init_logging(logger, f"{name.replace('.', '_')}.log")
    return logger


def connect_gcp():
    """
    Connect to GCP PostgreSQL database (home use only).
    Returns:
        con: dictionary with connect_string and engine
    """
    # Always use the same credentials and always grab gcp_server
    username = os.getenv('gcp_username')
    userpwd = os.getenv('gcp_password')
    server = os.getenv('gcp_server')
    #server = '67.172.197.116:5432'
    #server = '34.150.156.184:5432'

    con = {}
    db = 'optionsDB'
    connect_string = ''.join(['postgresql+psycopg2://', username, ':', userpwd, '@', server, '/', db])
    con['connect_string'] = connect_string
    con['engine'] = sa.create_engine(connect_string, echo=False)
    return con