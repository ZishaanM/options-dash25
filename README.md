# Options Dashboard

An intraday options analysis and prediction dashboard that combines historical pattern matching with Black-Scholes pricing to generate real-time trading recommendations.

## Purpose & Motivation

I asked my mentor about a project that would let me explore the intersection of data science, computer science, and quantitative finance. He proposed to me the project and the following hypothesis: **can intraday price patterns predict end-of-day returns?** I wanted to build a tool that could analyze real market data and produce actionable insights.

## The Problem

Traditional options pricing models like Black-Scholes assume constant volatility and don't account for intraday market dynamics. Meanwhile, traders often rely on intuition when deciding whether to buy or sell options during the trading day.

**The gap I identified:**
- Options tools calculate theoretical prices but don't contextualize them against likely price movements
- Pattern-based analysis exists for daily/weekly timeframes but rarely for intraday decisions
- There's no easy way to combine historical pattern similarity with options pricing for real-time recommendations

**My solution:** A dashboard that finds historically similar trading days, predicts where the stock will close, and uses those predictions to recommend whether specific options are worth buying—all updated in real-time.

## My Approach

### 1. Historical Pattern Matching
I implemented a similarity search algorithm that finds historical trading days with matching intraday return profiles:
- Return from market open
- Return from previous close  
- Return from intraday high/low
- Time of Day

The algorithm uses an adaptive threshold—starting strict and relaxing until it finds 30-100 similar patterns.

### 2. Statistical Prediction
From similar historical days, I calculate:
- Mean expected return to close
- 95% confidence interval using normal distribution
- Standard deviation for 1-sigma bounds

### 3. Options Pricing Integration
Using Black-Scholes (adjusted for intraday time-to-expiration), I calculate theoretical option prices and compare break-even points against the predicted price range to generate buy/sell signals.

## Technical Choices

| Technology | Why I Chose It |
|------------|----------------|
| **Python** | Rich ecosystem for quantitative finance (NumPy, SciPy, Pandas) |
| **PostgreSQL** | Handles millions of historical price records efficiently with SQL-based similarity queries |
| **Streamlit** | Rapid prototyping of interactive data apps—allowed me to focus on logic rather than frontend |
| **Plotly** | Interactive charts essential for financial visualization |
| **Black-Scholes** | Industry-standard options pricing; implemented from scratch to understand the math |

## Features

- **Real-time Pattern Matching**: Queries 15+ years of historical data to find similar trading days
- **Confidence Intervals**: 95% CI and 1-sigma bounds for predicted close price
- **Options Recommendations**: Color-coded buy signals based on break-even analysis
- **Greeks Calculation**: Delta, Gamma, and Probability ITM computed for each strike
- **Interactive Dashboard**: Three-tab interface for price charts, predictions, and detailed options tables

## Results

The dashboard successfully:
- Identifies 30-100 similar historical patterns within seconds
- Generates predictions with quantified uncertainty (confidence intervals)
- Produces clear buy/sell recommendations based on statistical edge


## Future Improvements

- [ ] Add live market data feed for real-time analysis (currently uses historical reference dates)
- [ ] Implement implied volatility calculation from actual option prices
- [ ] Backtest the recommendation system across multiple years
- [ ] Add more sophisticated similarity metrics (e.g., DTW for pattern matching)
- [ ] Support for multiple tickers simultaneously

---

## Technical Documentation

### Project Structure

```
options-dash25/
├── UI.py              # Streamlit dashboard interface
├── main.py            # Core analysis logic (pattern matching, predictions, Black-Scholes)
├── config.py          # Configuration (reference date/time)
├── z_util.py          # Utilities (logging, database connection)
├── Black_Scholes.py   # Standalone Black-Scholes pricing function
├── requirements.txt   # Python dependencies
├── log/               # Application logs
└── Dash-pics/         # Dashboard screenshots
```

### Installation

1. Clone the repository

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install streamlit plotly scipy
   ```

### Configuration

Set environment variables for database connectivity:

| Variable | Description |
|----------|-------------|
| `gcp_username` | PostgreSQL database username |
| `gcp_password` | PostgreSQL database password |
| `gcp_server` | Database server address (host:port) |

Edit `config.py` to set the analysis date and time:
```python
reference_date = '20100608'  # Format: YYYYMMDD
reference_time = 1200        # Format: HHMM (e.g., 1200 = 12:00 PM)
```

### Usage

**Run the Dashboard:**
```bash
streamlit run UI.py
```

**Run Analysis from Command Line:**
```bash
python main.py
```

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `find_sim_history()` | main.py | Adaptive pattern matching algorithm |
| `pred_ret()` | main.py | Statistical prediction with confidence intervals |
| `black_scholes_price()` | main.py | European option pricing |
| `calc_table_data()` | main.py | Generate options recommendation table |
| `calc_advanced_table_data()` | main.py | Detailed table with Greeks |

### Database Schema

The application expects a `returns` table:

| Column | Type | Description |
|--------|------|-------------|
| `date` | VARCHAR | Trading date (YYYYMMDD) |
| `time` | INT | Time of day (HHMM format) |
| `close` | FLOAT | Price at that time |
| `ret_from_open` | FLOAT | Return from market open |
| `ret_from_p_close` | FLOAT | Return from previous close |
| `ret_from_high` | FLOAT | Return from day's high |
| `ret_from_low` | FLOAT | Return from day's low |
| `ret_to_close` | FLOAT | Return to market close |

### Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- sqlalchemy >= 2.0.0
- psycopg2-binary >= 2.9.0
- streamlit
- plotly
- scipy
