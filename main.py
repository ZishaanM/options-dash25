import pandas as pd
import numpy as np
import scipy.stats as st
from datetime import datetime, time

import z_util as zu
from config import reference_date, reference_time

TABLE_NAME = 'returns'
logger = zu.get_logger(__name__)

# Default ticker - can be overridden when calling functions
DEFAULT_TICKER = "SPY"

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
    
    # Load data from parquet (cached)
    returns_df = zu.load_parquet(TABLE_NAME)
    
    # Extract reference values
    ret_from_open = current_day_returns[0]
    ret_from_p_close = current_day_returns[1]
    ret_from_high = current_day_returns[2]
    ret_from_low = current_day_returns[3]
    
    threshold = 0.001
    logger.info(f"Initial threshold: {threshold}")
    
    for i in range(100):
        # Filter by time
        df = returns_df[returns_df['time'] == curr_time].copy()
        
        # Apply threshold filters
        df = df[
            (abs(df['ret_from_open'] - ret_from_open) < threshold) &
            (abs(df['ret_from_p_close'] - ret_from_p_close) < threshold) &
            (abs(df['ret_from_high'] - ret_from_high) < threshold) &
            (abs(df['ret_from_low'] - ret_from_low) < threshold)
        ]
        
        # Calculate similarity scores
        df['diff_open'] = abs(df['ret_from_open'] - ret_from_open)
        df['diff_p_close'] = abs(df['ret_from_p_close'] - ret_from_p_close)
        df['diff_high'] = abs(df['ret_from_high'] - ret_from_high)
        df['diff_low'] = abs(df['ret_from_low'] - ret_from_low)
        df['total_diff'] = df['diff_open'] + df['diff_p_close'] + df['diff_high'] + df['diff_low']
        
        # Sort by total difference and limit to 100
        df = df.sort_values('total_diff').head(100)
        
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
    
    CI = st.norm.interval(0.95, loc=avg_ret_to_close, scale=std_ret_to_close)
    logger.info(f"95% Confidence interval: [{CI[0]:.4f}, {CI[1]:.4f}]")
    
    return avg_ret_to_close, CI, std_ret_to_close

def black_scholes_price(S, K, T_hours, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes price for a European option.

    Parameters:
    S : float
        Current price of the underlying asset
    K : float
        Strike price of the option
    T_hours : float
        Time to expiration in hours
    r : float
        Risk-free interest rate (annualized, as a decimal, e.g., 0.05 for 5%)
    sigma : float
        Volatility of the underlying asset (annualized, as a decimal)
    option_type : str
        'call' for call option, 'put' for put option

    Returns:
    float
        Theoretical price of the option
    """
    from math import log, sqrt, exp
    from scipy.stats import norm

    # Convert hours to years (assuming 252 trading days/year, 6.5 trading hours/day)
    trading_hours_per_year = 252 * 6.5
    T = T_hours / trading_hours_per_year

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def implied_vol():
    """
    Calculate the implied volatility of an option
    """
    return 0.32161

def calc_table_data(current_price, option_switch, lower_price, upper_price, lower_1sigma_price, upper_1sigma_price):
    """
    Calculate options data and return a complete table
    """
    rows = []
    S = current_price
    K = round(S - 10)
    
    # Calculate time to expiry in minutes (HHMM format conversion)
    end_time = 1600
    end_hrs, end_mins = end_time // 100, end_time % 100
    ref_hrs, ref_mins = reference_time // 100, reference_time % 100
    TOE_minutes = (end_hrs - ref_hrs) * 60 + (end_mins - ref_mins)
    TOE_hours = TOE_minutes / 60  # Convert to hours for Black-Scholes
    
    r = 0.03961
    sigma = implied_vol()
    option_type = "call" if option_switch else "put"
    
    for i in range(15):        
        price = black_scholes_price(S, K, TOE_hours, r, sigma, option_type)
        buy_recommendation = ""
        if option_type == "call":
            intrinsic_value = max(0, S - K - price)
            break_even = K + price
            if break_even < lower_price:
                buy_recommendation = "STRONG YES"
            elif break_even < lower_1sigma_price:
                buy_recommendation = "Yes"
            elif break_even < upper_1sigma_price:
                buy_recommendation = "No"
            elif break_even < upper_price:
                buy_recommendation = "No"
            else:
                buy_recommendation = "STRONG NO"
        else:  # put
            intrinsic_value = max(0, K - S - price)
            break_even = K - price
            if break_even < lower_price:
                buy_recommendation = "STRONG NO"
            elif break_even < lower_1sigma_price:
                buy_recommendation = "No"
            elif break_even < upper_1sigma_price:
                buy_recommendation = "No"
            elif break_even < upper_price:
                buy_recommendation = "Yes"
            else:
                buy_recommendation = "STRONG YES"
        rows.append({
            "Strike": K,
            "Type": option_type.capitalize(),
            "Time to Expiration": f"{TOE_minutes:.0f} minutes",
            "Theoretical Price": f"${price:.2f}",
            "Buy?": buy_recommendation,
            "Break-even": f"${break_even:.2f}"
        })
        K+=2.5
    
    df = pd.DataFrame(rows)
    
    # Apply row-wise styling based on Buy? column
    def style_rows(row):
        buy_value = row['Buy?']
        if buy_value == "STRONG YES":
            return ['background-color: #006400; color: white;'] * len(row)  # dark green
        elif buy_value == "Yes":
            return ['background-color: #90ee90; color: black;'] * len(row)  # light green
        elif buy_value == "No":
            return ['background-color: #ffcccb; color: black;'] * len(row)  # light red
        elif buy_value == "STRONG NO":
            return ['background-color: #b22222; color: white;'] * len(row)  # firebrick/dark red
        else:
            return [''] * len(row)  # no styling
    
    styled_df = df.style.apply(style_rows, axis=1)
    return styled_df

def calc_advanced_table_data(current_price, option_switch, lower_price, upper_price, lower_1sigma_price, upper_1sigma_price):
    """
    Calculate advanced options data with Greeks and additional metrics
    """
    from math import log, sqrt, exp
    from scipy.stats import norm
    
    rows = []
    S = current_price
    K = round(S - 10)
    
    # Calculate time to expiry in minutes (HHMM format conversion)
    end_time = 1600
    end_hrs, end_mins = end_time // 100, end_time % 100
    ref_hrs, ref_mins = reference_time // 100, reference_time % 100
    TOE_minutes = (end_hrs - ref_hrs) * 60 + (end_mins - ref_mins)
    TOE_hours = TOE_minutes / 60  # Convert to hours for Black-Scholes
    
    r = 0.03961
    sigma = implied_vol()
    option_type = "call" if option_switch else "put"
    
    # Convert hours to years for Greeks calculation
    trading_hours_per_year = 252 * 6.5
    T = TOE_hours / trading_hours_per_year
    
    for i in range(15):        
        price = black_scholes_price(S, K, TOE_hours, r, sigma, option_type)
        
        # Calculate Greeks if T > 0
        if T > 0 and sigma > 0:
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            
            # Delta
            if option_type == "call":
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
            
            # Probability ITM
            if option_type == "call":
                prob_itm = norm.cdf(d2)
            else:
                prob_itm = norm.cdf(-d2)
        else:
            delta = gamma = prob_itm = 0
        
        # Buy logic using price bounds
        if option_type == "call":
            intrinsic_value = max(0, S - K)
            break_even = K + price
            if break_even < lower_price:
                buy_recommendation = "STRONG YES"
            elif break_even < lower_1sigma_price:
                buy_recommendation = "Yes"
            elif break_even < upper_1sigma_price:
                buy_recommendation = "No"
            elif break_even < upper_price:
                buy_recommendation = "No"
            else:
                buy_recommendation = "STRONG NO"
        else:  # put
            intrinsic_value = max(0, K - S)
            break_even = K - price
            if break_even < lower_price:
                buy_recommendation = "STRONG NO"
            elif break_even < lower_1sigma_price:
                buy_recommendation = "No"
            elif break_even < upper_1sigma_price:
                buy_recommendation = "No"
            elif break_even < upper_price:
                buy_recommendation = "Yes"
            else:
                buy_recommendation = "STRONG YES"
        
        rows.append({
            "Strike": K,
            "Type": option_type.capitalize(),
            "Time to Expiration": f"{TOE_minutes:.0f} minutes",
            "Bid": "Fake",
            "Ask": "Fake",
            "Volume": "Fake",
            "Implied Vol": f"{sigma*100:.1f}%",
            "Theoretical Price": f"${price:.2f}",
            "Delta": f"{delta:.3f}",
            "Gamma": f"{gamma:.4f}",
            "STD": f"{sigma:.3f}",
            "Buy?": buy_recommendation,
            "Probability ITM @ Exp.": f"{prob_itm*100:.1f}%",
            "ATM price": f"${S:.2f}"
        })
        K = K + 2.5
    
    df = pd.DataFrame(rows)
    
    # Apply row-wise styling based on Buy? column
    def style_rows(row):
        buy_value = row['Buy?']
        if buy_value == "STRONG YES":
            return ['background-color: #018E01; color: white;'] * len(row)  # dark green
        elif buy_value == "Yes":
            return ['background-color: #59CC59; color: white;'] * len(row)  # light green
        elif buy_value == "No":
            return ['background-color: #FE6667; color: white;'] * len(row)  # light red
        elif buy_value == "STRONG NO":
            return ['background-color: #B10808; color: white;'] * len(row)  # firebrick/dark red
        else:
            return [''] * len(row)  # no styling
    
    styled_df = df.style.apply(style_rows, axis=1)
    return styled_df

if __name__ == "__main__":
    logger.info("Starting main execution")
    
    # Use the default ticker
    ticker = DEFAULT_TICKER
    logger.info(f"Analyzing {ticker} for date {reference_date} at time {reference_time}")
    
    try:
        # Load returns data from parquet and filter (convert types to match parquet data)
        returns_df = zu.load_parquet(TABLE_NAME)
        ref_date = int(reference_date) if isinstance(reference_date, str) else reference_date
        ref_time = int(reference_time) if isinstance(reference_time, str) else reference_time
        current_day = returns_df[(returns_df['date'] == ref_date) & (returns_df['time'] == ref_time)].copy()
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
        logger.error("Make sure the parquet files exist in the script directory")
        print(f"Error running find_sim_history: {e}")
        print("Make sure the parquet files exist in the script directory")


    #for loop to run through 100 dates, and find how many times the actual return is within the CI
    #loop through thresholds to find which one gives the best results?
    #correlation coeff between similarity in other categories and ret_to_close