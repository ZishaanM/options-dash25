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