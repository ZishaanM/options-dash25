import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from datetime import time as dt_time

import z_util as zu
from main import find_sim_history, OLD_pred_ret, pred_ret, convert_time_to_datetime, calc_table_data, calc_advanced_table_data
from config import reference_date, reference_time
from main import get_today_data
from main import load_data

# Initialize logger
logger = zu.get_logger(__name__)
st.title("Options Dashboard")
tab1, tab2, tab3 = st.tabs(["Home", "Data", "Adv. Table"])
with tab1:
   
    st.header("Price Graph")
    st.text("Price of reference day until reference time")
    realtime_toggle = st.toggle("Realtime", value=True)
    month = datetime.now().month
    day = datetime.now().day
    dayofweek = datetime.now().weekday()
    year = datetime.now().year
    time = datetime.now().time().hour * 100 + datetime.now().time().minute
    if realtime_toggle:
        user_ticker = st.text_input("Ticker", value="SPY", max_chars=10)
        ticker = user_ticker.upper() if user_ticker else "SPY"
        get_today_data(ticker)
    else:
        date_col1, date_col2, date_col3, date_col4, date_col5 = st.columns(5)
        with date_col1:
            user_ticker = st.text_input("Ticker", value="SPY", max_chars=10)
        with date_col2:
            month = st.selectbox("Month", options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
            month_dict = {
                "January": 1,
                "February": 2,
                "March": 3,
                "April": 4,
                "May": 5,
                "June": 6,
                "July": 7,
                "August": 8,
                "September": 9,
                "October": 10,
                "November": 11,
                "December": 12
            }
            month = month_dict[month]
        with date_col3:
            day = int(st.number_input("Day", value=2, min_value=1, max_value=31, step=1))
        with date_col4:
            year = st.number_input("Year", value=2015, min_value=1999, max_value=2020, step=1)
        with date_col5:
            #it needs to skip 60-99
            selected_time = st.time_input("Reference Time", value=dt_time(13, 00), step=60)
            time = selected_time.hour * 100 + selected_time.minute
        reference_date = f"{year}{month:02d}{day:02d}"
        ticker = user_ticker.upper() if user_ticker else "SPY"
        dayofweek = datetime(year, month, day).weekday()
        returns_df_check = load_data()
        if returns_df_check[(returns_df_check['date'] == int(reference_date))].empty or time < 930 or time > 1600 or dayofweek > 4:
            st.write("Market is closed or data not available, please select a past date/time to simulate")
    
    

    

current_day = None
similar_history = None
logger.info("vars initialized to None")

# Set reference_date for realtime mode to today's date
if realtime_toggle:
    today = datetime.now()
    reference_date = f"{today.year}{today.month:02d}{today.day:02d}"
    
formatted_reference_date = pd.to_datetime(reference_date, format='%Y%m%d').strftime('%B %d, %Y')  # Time as integer (1300 = 1:00 PM)
logger.info("params initialized")
# Create option toggle outside the try block
col1, col2, col3 = st.columns([4, 1, 1])
with col2:
    st.write(f"Ticker: {ticker}")
with col3:
    option_type = st.selectbox(
        "Option Type", 
        options=["Call", "Put"],
        index=0,
        help="Select Call or Put options"
    )
option_switch = option_type == "Call"

# Create empty table placeholders that will be populated later
summ_table_placeholder = None

with tab3:
    st.subheader("Advanced Table")
    empty_adv_table = pd.DataFrame(columns=["Strike", "Type", "Time to Expiration", "Bid", "Ask", "Volume", "Implied Vol", "Premium", "Delta", "Gamma", "STD", "Buy?", "Probability ITM @ Exp.", "ATM price"])
    adv_table_placeholder = st.empty()
    adv_table_placeholder.dataframe(empty_adv_table, use_container_width=True)

try:
    # Load data from parquet
    logger.info("Loading data from parquet...")
    returns_df = zu.load_parquet('returns')
    
    # Get current day data (convert types to match parquet data)
    logger.info("Fetching current day data...")
    ref_date = int(reference_date) if isinstance(reference_date, str) else reference_date
    ref_time = int(reference_time) if isinstance(reference_time, str) else reference_time
    if realtime_toggle:
        current_day = get_today_data(ticker)
    else:
        current_day = returns_df[(returns_df['date'] == ref_date) & (returns_df['time'] <= ref_time)].copy()
    
    if current_day.empty:
        logger.error("No data found for the reference date and time")
        logger.info(f"Searched for: date='{reference_date}' AND time='{reference_time}'")
        st.stop()
    
    # Add datetime column to current_day if it doesn't exist
    if 'datetime' not in current_day.columns:
        current_day['datetime'] = current_day.apply(lambda row: convert_time_to_datetime(str(row['date']), int(row['time'])), axis=1)
    else:
        # For realtime data, ensure datetime is timezone-naive for consistency
        if current_day['datetime'].dt.tz is not None:
            current_day['datetime'] = current_day['datetime'].dt.tz_localize(None)
    
    logger.info(f"Found {len(current_day)} records for {reference_date} at {reference_time}")
    
    logger.info("Finding similar historical patterns...")
    similar_history, threshold = find_sim_history(current_day, ticker)
    
    if similar_history is not None and not similar_history.empty:
        st.success(f"Found {len(similar_history)} similar historical patterns")
        current_day_sorted = current_day.sort_values('datetime')
        
        market_open = convert_time_to_datetime(reference_date, 930)
        current_day_sorted = current_day_sorted[current_day_sorted['datetime'] >= market_open]
        
        # Create Plotly figure
        fig = go.Figure()
        m_color = 'blue'
        if current_day_sorted['close'].iloc[-1] > current_day_sorted['close'].iloc[0]:
            l_color = 'green'
        else:
            l_color = 'red'

        # Add line plot
        fig.add_trace(go.Scatter(
            x=current_day_sorted['datetime'],
            y=current_day_sorted['close'],
            mode='lines+markers',
            name='Open Price',
            line=dict(color=f'{l_color}', width=2),
            marker=dict(size=4)
        ))
        
        avg_ret, confidence_interval, std_ret = OLD_pred_ret(similar_history)
        
        # Get current price to convert returns to price levels
        current_price = current_day_sorted['close'].iloc[-1]  # Latest price
        
        # Convert return predictions to actual price predictions
        predicted_price = current_price * (1 + avg_ret)
        lower_price = current_price * (1 + confidence_interval[0])
        upper_price = current_price * (1 + confidence_interval[1])
        diff = upper_price - lower_price

        fig.add_hline(y=current_day_sorted['close'].iloc[0], line_color=f'{l_color}', line_width=2, line_dash='dash')
        
        # Set x-axis limits for market hours (9:30 AM to 4:00 PM)
        start_time = convert_time_to_datetime(reference_date, 930)
        current_time = convert_time_to_datetime(reference_date, reference_time)
        end_time = convert_time_to_datetime(reference_date, 1600)

        # Alternative: Add special marker points at the end of the chart for CI levels
        end_time_marker = end_time
        fig.add_trace(go.Scatter(
            x=[end_time_marker, end_time_marker],
            y=[lower_price, upper_price],
            mode='markers',
            marker=dict(
                #symbol='hash',
                symbol=41,
                size=8,
                color=m_color,
                line=dict(width=2, color=f'{m_color}')
            ),
            name='95% CI Bounds',
            showlegend=True
        ))
        # Calculate 1 sigma bounds around the predicted return
        upper_1sigma = current_price * (1 + avg_ret + std_ret)
        lower_1sigma = current_price * (1 + avg_ret - std_ret)
        
        fig.add_trace(go.Scatter(
            x=[end_time_marker, end_time_marker],
            y=[lower_1sigma, upper_1sigma],
            mode='markers',
            marker=dict(
                #symbol='hash',
                symbol=41,
                size=8,
                color=m_color,
                line=dict(width=2, color=f'{m_color}')
            ),
            name='1 Sigma',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[end_time_marker],
            y=[predicted_price],
            mode='markers',
            marker=dict(
                symbol='arrow-left',
                size=12,
                color=m_color,
                line=dict(width=2, color=f'dark{m_color}')
                #line=dict(width=2, color='blue')
            ),
            name='predicted price',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_price],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=8,
                color=l_color,
                line=dict(width=2, color=f'{l_color}')
            ),
            name='Current Price',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Intraday Price Movement on {formatted_reference_date}',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            width=1000,
            height=500,
            showlegend=True,
            legend=dict(
                x=0,
                y=1
            ),
            hovermode='x unified'
            #,
            # yaxis=dict(
            #     #range=[lower_price + diff*(.5), upper_price + diff*(.5)],
            #     tickmode='array',
            #     tickvals=[lower_price, upper_price],
            #     ticktext=[f'95% CI: {lower_price:.2f}', f'95% CI: {upper_price:.2f}'],
            #     side='right'
            # )
        )
        
        # Format x-axis
        fig.update_xaxes(
            range=[start_time, end_time + pd.Timedelta(minutes=10)],
            tickformat='%H:%M',
            dtick=30*60*1000,  # 30 minutes in milliseconds
            gridcolor='lightgray',
            gridwidth=0.5
        )
        
        # Format y-axis
        fig.update_yaxes(
            gridcolor='lightgray',
            gridwidth=0.5
        )

        with tab1:
            st.plotly_chart(fig, use_container_width=True)
            
            # Create and populate the summary table
            st.markdown("---")
            st.subheader("Summary Table")
            populated_summ_table = calc_table_data(similar_history, current_price, option_switch, lower_price, upper_price, lower_1sigma, upper_1sigma)
            st.dataframe(populated_summ_table, use_container_width=True)
        
        with tab3:
            # Populate the advanced table with actual data
            populated_adv_table = calc_advanced_table_data(similar_history, current_price, option_switch, lower_price, upper_price, lower_1sigma, upper_1sigma)
            adv_table_placeholder.dataframe(populated_adv_table, use_container_width=True)

        with tab2:
            st.header("Prediction Results")
            
            st.write(f"**Similar Historical Patterns Found:** {len(similar_history)}")
            st.write(f"**Threshold Used:** {threshold:.4f}")
            st.write(f"**Predicted Return to Close:** {avg_ret:.4f} ({avg_ret*100:.2f}%) = ${current_price * (1 + avg_ret):.4f}")
            st.write(f"**95% Confidence Interval:** ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}) = (${lower_price:.4f}, ${upper_price:.4f})")
            lower_1sigma = current_price * (1 + avg_ret - std_ret)
            upper_1sigma = current_price * (1 + avg_ret + std_ret)
            st.write(f"**1 Sigma:** ({std_ret:.4f}) = (${lower_1sigma:.4f}, ${upper_1sigma:.4f})")
    
            # Show actual vs predicted if available
            if 'ret_to_close' in current_day.columns:
                actual_ret = current_day['ret_to_close'].iloc[0]
                st.write(f"**Actual Return to Close:** {actual_ret:.4f} ({actual_ret*100:.2f}%)")
                st.write(f"**Prediction Error:** {abs(actual_ret - avg_ret):.4f} ({abs(actual_ret - avg_ret)*100:.2f}%)")
    else:
        logger.warning("No similar historical patterns found")
        
except Exception as e:
    logger.error(f"Error loading data: {e}")
    logger.error("Make sure the parquet files exist in the script directory")
    
    # Show debug information
    st.error(f"Error: {e}")
    st.info("Debug Information")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_files = ['returns.parquet', 'av_minute.parquet']
    for pf in parquet_files:
        filepath = os.path.join(script_dir, pf)
        if os.path.exists(filepath):
            st.write(f"✅ {pf}: Found")
        else:
            st.write(f"❌ {pf}: Not found")
    
    # Show the full error traceback
    import traceback
    st.code(traceback.format_exc())
