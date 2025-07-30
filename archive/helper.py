import pandas as pd
import numpy as np

import z_util as zu


def get_current_day_data(current_day: pd.DataFrame,
                    ticker: str) -> pd.DataFrame:
    """
    Get the current day's data for a given ticker
    Args:
        current_day: pandas dataframe with columns: date, time, close
        ticker: string with the ticker to get the data for
    Returns:
        df: pandas dataframe with columns: date, time, close
    """
    df = pd.read_sql(f"SELECT close FROM {ticker} WHERE date = {current_day}")
    return df.head()


def get_open_price(current_day: pd.Datetime, ticker: str) -> float:
    query = f"SELECT open FROM {ticker} WHERE date = '{current_day}' AND time = '09:30:00'"
    result = pd.read_sql(query)
    if not result.empty:
        return result.iloc[0]['open']
    else:
        return np.nan


def get_previous_close_price(current_day: pd.Timestamp, ticker: str) -> float:
    """
    Get the previous day's close price for a given ticker.
    Args:
        current_day: pandas Timestamp representing the current day
        ticker: string with the ticker to get the data for
    Returns:
        float: previous day's close price, or np.nan if not found
    """
    # Find the previous trading day
    prev_day = zu.get_previous_trading_day(current_day)
    query = f"SELECT close FROM {ticker} WHERE date = '{prev_day}' AND time = '16:00:00'"
    result = pd.read_sql(query)
    if not result.empty:
        return result.iloc[0]['close']
    else:
        return np.nan
    

