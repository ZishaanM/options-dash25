# Options Dashboard

An intraday options analysis and prediction dashboard built with Streamlit. This application uses historical pattern matching to predict end-of-day stock returns and provides options trading recommendations using Black-Scholes pricing.

## Overview

The dashboard analyzes intraday price movements and finds historically similar trading patterns to predict how the stock will close. It then uses these predictions along with Black-Scholes option pricing to recommend whether to buy or sell options at various strike prices.

## Features

- **Pattern Matching**: Finds similar historical trading days based on:
  - Return from open
  - Return from previous close
  - Return from high
  - Return from low

- **Return Prediction**: Calculates predicted return to close with:
  - Average expected return
  - 95% confidence interval
  - Standard deviation (1-sigma bounds)

- **Black-Scholes Pricing**: Theoretical option pricing with Greeks:
  - Delta
  - Gamma
  - Probability ITM at expiration

- **Trading Recommendations**: Color-coded buy/sell signals based on whether option break-even prices fall within predicted price ranges

## Project Structure

```
options-dash25/
├── UI.py              # Streamlit dashboard interface
├── main.py            # Core analysis logic (pattern matching, predictions, Black-Scholes)
├── config.py          # Configuration (reference date/time)
├── z_util.py          # Utilities (logging, database connection)
├── Black_Scholes.py   # Standalone Black-Scholes pricing function
├── requirements.txt   # Python dependencies
├── log/               # Application logs
├── Dash-pics/         # Dashboard screenshots
└── archive/           # Legacy/archived code
```

## Installation

1. Clone the repository

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional Streamlit dependencies:
   ```bash
   pip install streamlit plotly scipy
   ```

## Configuration

### Environment Variables

Set the following environment variables for database connectivity:

| Variable | Description |
|----------|-------------|
| `gcp_username` | PostgreSQL database username |
| `gcp_password` | PostgreSQL database password |
| `gcp_server` | Database server address (host:port) |

### Reference Date/Time

Edit `config.py` to set the analysis date and time:

```python
reference_date = '20100608'  # Format: YYYYMMDD
reference_time = 1200        # Format: HHMM (e.g., 1200 = 12:00 PM)
```

## Usage

### Running the Dashboard

```bash
streamlit run UI.py
```

The dashboard will open in your browser with three tabs:

1. **Home**: Intraday price chart with predicted close range and confidence intervals
2. **Data**: Detailed prediction statistics and comparison with actual returns
3. **Adv. Table**: Advanced options table with Greeks and probability metrics

### Running Analysis from Command Line

```bash
python main.py
```

This runs the pattern matching and prediction analysis, outputting results to the console.

## Dashboard Tabs

### Home Tab
- Interactive price chart showing intraday movement
- Predicted price at close (arrow marker)
- 95% confidence interval bounds
- 1-sigma bounds
- Summary table with options recommendations

### Data Tab
- Predicted return to close
- 95% confidence interval
- Number of similar historical patterns found
- Threshold used for pattern matching
- Actual vs predicted comparison (when available)

### Advanced Table Tab
- Strike prices with theoretical Black-Scholes prices
- Greeks (Delta, Gamma)
- Implied volatility
- Probability ITM at expiration
- Color-coded buy recommendations:
  - 🟢 **Dark Green**: STRONG YES
  - 🟢 **Light Green**: Yes
  - 🔴 **Light Red**: No
  - 🔴 **Dark Red**: STRONG NO

## How It Works

1. **Data Retrieval**: Fetches intraday return data from the PostgreSQL database for the reference date

2. **Similarity Search**: Queries historical data to find days with similar intraday return patterns using an adaptive threshold algorithm

3. **Prediction**: Calculates mean return-to-close and confidence intervals from similar historical days

4. **Options Pricing**: Uses Black-Scholes model to calculate theoretical option prices, adjusting for intraday time-to-expiration

5. **Recommendations**: Compares option break-even prices against predicted price ranges to generate buy/sell signals

## Key Functions

### `main.py`
- `find_sim_history()` - Pattern matching algorithm
- `pred_ret()` - Return prediction with confidence intervals
- `black_scholes_price()` - Option pricing
- `calc_table_data()` - Summary options table
- `calc_advanced_table_data()` - Detailed options table with Greeks

### `z_util.py`
- `get_logger()` - Configured logging with file rotation
- `connect_gcp()` - GCP PostgreSQL database connection

## Database Schema

The application expects a `returns` table with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | VARCHAR | Trading date (YYYYMMDD) |
| `time` | INT | Time of day (HHMM format) |
| `close` | FLOAT | Close price at that time |
| `ret_from_open` | FLOAT | Return from market open |
| `ret_from_p_close` | FLOAT | Return from previous day's close |
| `ret_from_high` | FLOAT | Return from day's high |
| `ret_from_low` | FLOAT | Return from day's low |
| `ret_to_close` | FLOAT | Return from this time to market close |

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- sqlalchemy >= 2.0.0
- psycopg2-binary >= 2.9.0
- streamlit
- plotly
- scipy

## Logging

Logs are written to:
- `log/main.log` - Main application logs (rotating, 1MB max)
- `cron.log` - Error-level logs for monitoring

## License

This project is for personal/educational use.



