import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
import scipy.stats as st
from datetime import datetime, time

import z_util as zu


def convert_time_to_datetime(date_str: str, time_int: int) -> pd.Timestamp:
    """
    Convert date string and time integer to pandas datetime
    Args:
        date_str: Date in format 'YYYYMMDD'
        time_int: Time in format HHMM (e.g., 930 for 9:30, 1500 for 15:00)
    Returns:
        pd.Timestamp object
    """
    # Convert time integer to hours and minutes
    hours = int(time_int // 100)
    minutes = int(time_int % 100)
    
    # Parse date string
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    return pd.Timestamp(year=year, month=month, day=day, hour=hours, minute=minutes)



table = 'returns'
logger = zu.get_logger(__name__)

reference_date = '20071019'
time = 1200

def find_sim_history(reference_day: pd.DataFrame,
                     ticker: str) -> pd.DataFrame:
    """
    Find history that is most similar to the current day's price pattern
    Args:
        df: pandas dataframe with columns: date, time, close
    Returns:
        df: pandas dataframe with columns: date, time, close
    """
    logger.info(f"Starting similarity search for ticker: {ticker}")
    
    table_row = ["ret_from_open", "ret_from_p_close", "ret_from_high", "ret_from_low", "ret_to_close"]
    current_day_returns = [float(reference_day[row].iloc[0]) if row in reference_day.columns and not pd.isna(reference_day[row].iloc[0]) else None for row in table_row]
    curr_time = int(reference_day['time'].iloc[0])  # Convert to native Python int
    
    logger.info(f"Current time: {curr_time}, Returns: {current_day_returns}")
    
    engine = zu.connect_gcp()['engine']
    threshold = 0.001
    logger.info(f"Initial threshold: {threshold}")
    params = {
        "curr_time": curr_time,
        "ret_from_open": current_day_returns[0],
        "ret_from_p_close": current_day_returns[1],
        "ret_from_high": current_day_returns[2],
        "ret_from_low": current_day_returns[3]
    }
    for i in range(100):
        query = text(f"""
            SELECT date, time, close, ret_from_open, ret_from_p_close, ret_from_high, ret_from_low, ret_to_close,
                -- Calculate similarity score for each component (excluding ret_to_close)
                ABS(ret_from_open - :ret_from_open) as diff_open,
                ABS(ret_from_p_close - :ret_from_p_close) as diff_p_close,
                ABS(ret_from_high - :ret_from_high) as diff_high,
                ABS(ret_from_low - :ret_from_low) as diff_low
            FROM {table}
            WHERE time = :curr_time
            AND ABS(ret_from_open - :ret_from_open) < {threshold}
            AND ABS(ret_from_p_close - :ret_from_p_close) < {threshold}
            AND ABS(ret_from_high - :ret_from_high) < {threshold}
            AND ABS(ret_from_low - :ret_from_low) < {threshold}
            ORDER BY (
                ABS(ret_from_open - :ret_from_open) + 
                ABS(ret_from_p_close - :ret_from_p_close) + 
                ABS(ret_from_high - :ret_from_high) + 
                ABS(ret_from_low - :ret_from_low)
            ) ASC
            LIMIT 100  -- Get top 100 most similar days
        """)
        df = pd.read_sql(query, engine, params=params)
        logger.info(f"Iteration {i+1}: Found {len(df)} similar patterns with threshold {threshold}")
        
        if len(df) < 30:
            threshold += 0.001
            logger.info(f"Increasing threshold to {threshold}")
        else:
            logger.info(f"Found sufficient patterns ({len(df)}) with threshold {threshold}")
            break

    logger.info(f"Similarity search completed. Final threshold: {threshold}, Patterns found: {len(df)}")
    
    # Add datetime column to the result
    if not df.empty:
        df['datetime'] = df.apply(lambda row: convert_time_to_datetime(str(row['date']), int(row['time'])), axis=1)
    
    return df, threshold


def pred_ret(similar_history: pd.DataFrame) -> float:
    """
    Predict the return of the current day based on the similar historical patterns
    """
    
    logger.info(f"Calculating predictions from {len(similar_history)} similar patterns")
    
    avg_ret_to_close = similar_history["ret_to_close"].mean()
    std_ret_to_close = similar_history["ret_to_close"].std()
    
    logger.info(f"Average return to close: {avg_ret_to_close:.4f}, Std dev: {std_ret_to_close:.4f}")
    
    #z_score = (current_day_returns[4] - avg_ret_to_close) / std_ret_to_close
    #p_value = st.norm.cdf(z_score)
    CI = st.norm.interval(0.95, loc=avg_ret_to_close, scale=std_ret_to_close)
    logger.info(f"95% Confidence interval: [{CI[0]:.4f}, {CI[1]:.4f}]")
    
    return avg_ret_to_close, CI, std_ret_to_close


if __name__ == "__main__":
    logger.info("Starting main execution")
    
    # Use the module-level variables (already defined above)
    ticker = "SPY"  # Example ticker
    logger.info(f"Analyzing {ticker} for date {reference_date} at time {time}")
    
    try:
        current_day = pd.read_sql(f"SELECT * FROM returns WHERE date='{reference_date}' AND time='{time}'", zu.connect_gcp()['engine'])
        logger.info(f"Retrieved current day data: {len(current_day)} rows")
        # Find similar historical patterns
        similar_history, threshold = find_sim_history(current_day, ticker)
        prediction = pred_ret(similar_history)
        
        # Extract prediction components
        avg_ret, confidence_interval, std_ret = prediction
        actual_ret = current_day['ret_to_close'].iloc[0]
        z_score = (actual_ret - avg_ret) / std_ret if std_ret != 0 else 0
        
        logger.info(f"Analysis complete - Predicted: {avg_ret:.4f}, Actual: {actual_ret:.4f}, Z-score: {z_score:.2f}")
        
        # Print results in a clean format
        print("=" * 60)
        print(f"SIMILARITY ANALYSIS FOR {ticker}")
        print("=" * 60)
        print(f"Reference Date: {reference_date} at {time}")
        print(f"Similar Historical Patterns Found: {len(similar_history)}")
        print()
        
        print("PREDICTION RESULTS:")
        print("-" * 30)
        print(f"Predicted Return to Close: {avg_ret:.4f} ({avg_ret*100:.2f}%)")
        print(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        print(f"                        [{confidence_interval[0]*100:.2f}%, {confidence_interval[1]*100:.2f}%]")
        print(f"Standard Deviation: {std_ret:.4f}")
        print(f"Range: {confidence_interval[1]-confidence_interval[0]:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print()
        
        print("ACTUAL vs PREDICTED:")
        print("-" * 30)
        print(f"Actual Return to Close: {actual_ret:.4f} ({actual_ret*100:.2f}%)")
        print(f"Prediction Error: {abs(actual_ret - avg_ret):.4f} ({abs(actual_ret - avg_ret)*100:.2f}%)")
        print(f"Z-Score: {z_score:.2f}")
        
        # Check if actual is within confidence interval
        within_ci = confidence_interval[0] <= actual_ret <= confidence_interval[1]
        print(f"Within 95% CI: {'✓ YES' if within_ci else '✗ NO'}")
        print()
        
        print("TOP 5 MOST SIMILAR HISTORICAL DAYS:")
        print("-" * 50)
        display_cols = ['date', 'ret_from_open', 'ret_from_p_close', 'ret_from_high', 'ret_from_low', 'ret_to_close']
        if all(col in similar_history.columns for col in display_cols):
            similar_display = similar_history[display_cols].head()
            # Format the dataframe for better display
            for col in ['ret_from_open', 'ret_from_p_close', 'ret_from_high', 'ret_from_low', 'ret_to_close']:
                if col in similar_display.columns:
                    similar_display[col] = similar_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            print(similar_display.to_string(index=False))
        else:
            print(similar_history.head().to_string(index=False))
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running find_sim_history: {e}")
        logger.error("Make sure your environment variables are set and database is accessible")
        print(f"Error running find_sim_history: {e}")
        print("Make sure your environment variables are set and database is accessible")


    #for loop to run through 100 dates, and find how many times the actual return is within the CI
    #loop through thresholds to find which one gives the best results?
    #correlation coeff between similarity in other categories and ret_to_close