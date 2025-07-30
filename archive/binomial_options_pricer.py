import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.stats import binom, norm
import math

S = 100
K = 100
T = 1  # Time to expiration in years (e.g., 1 year)
r = 0.05
sigma = 0.2
n = 10 # Number of time steps in the binomial tree
option_type = 'call'


def calculate_tree_size(n: int) -> int:
    """
    Calculate the total number of nodes in a recombining binomial tree.
    
    Args:
        n: Number of time steps
    
    Returns:
        Total number of nodes
    """
    return (n + 1) * (n + 2) // 2

def binomial_tree_parameters(S: float, K: float, T: float, r: float, sigma: float, n: int) -> Tuple[float, float, float, float]:
    """
    Calculate the parameters for the binomial tree:
    - u: up factor
    - d: down factor
    - p: risk-neutral probability
    - dt: time step
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        n: Number of time steps
    
    Returns:
        Tuple of (u, d, p, dt)
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    return u, d, p, dt

def build_stock_price_tree(S: float, u: float, d: float, n: int) -> np.ndarray:
    """
    Build the stock price tree using a recombining structure.
    For n time steps, we need (n+1)(n+2)/2 nodes total.
    
    Args:
        S: Initial stock price
        u: Up factor
        d: Down factor
        n: Number of time steps
    
    Returns:
        Stock price tree as numpy array
    """
    tree = np.zeros((n + 1, n + 1))
    tree[0, 0] = S
    
    for i in range(1, n + 1):
        for j in range(i + 1):
            # In a recombining tree, the number of up and down moves
            # determines the price, not the order of moves
            tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    return tree

def calculate_option_payoff(tree: np.ndarray, K: float, option_type: str = 'call') -> np.ndarray:
    """
    Calculate the option payoff at expiration.
    
    Args:
        tree: Stock price tree
        K: Strike price
        option_type: 'call' or 'put'
    
    Returns:
        Option payoff tree
    """
    n = tree.shape[1] - 1
    payoff = np.zeros_like(tree)
    
    if option_type.lower() == 'call':
        payoff[:, n] = np.maximum(tree[:, n] - K, 0)
    else:  # put
        payoff[:, n] = np.maximum(K - tree[:, n], 0)
    
    return payoff

def backward_induction(payoff: np.ndarray, p: float, r: float, dt: float) -> np.ndarray:
    """
    Perform backward induction to calculate option prices.
    
    Args:
        payoff: Option payoff tree
        p: Risk-neutral probability
        r: Risk-free rate
        dt: Time step
    
    Returns:
        Option price tree
    """
    n = payoff.shape[1] - 1
    option_tree = payoff.copy()
    
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + 
                                                  (1 - p) * option_tree[j + 1, i + 1])
    
    return option_tree

def plot_binomial_tree(stock_tree: np.ndarray, option_tree: np.ndarray, 
                      title: str = "Binomial Tree Visualization"):
    """
    Plot the binomial tree visualization with improved clarity.
    
    Args:
        stock_tree: Stock price tree
        option_tree: Option price tree
        title: Plot title
    """
    n = stock_tree.shape[1] - 1
    initial_option_price = option_tree[0, 0]  # Price you paid for the option
    
    plt.figure(figsize=(15, 10))
    
    # Plot stock prices
    for i in range(n + 1):
        for j in range(i + 1):
            plt.scatter(i, stock_tree[j, i], color='blue', s=100)
            # Format the text to show stock price, option value, and potential profit
            if i == n:  # At expiration
                potential_profit = option_tree[j, i] - initial_option_price
                plt.text(i, stock_tree[j, i], 
                        f'S=${stock_tree[j, i]:.2f}\nV=${option_tree[j, i]:.2f}\nProfit=${potential_profit:.2f}',
                        ha='center', va='bottom')
            else:  # Earlier nodes
                plt.text(i, stock_tree[j, i], 
                        f'S=${stock_tree[j, i]:.2f}\nV=${option_tree[j, i]:.2f}',
                        ha='center', va='bottom')
    
    # Connect nodes with different colors for up and down moves
    for i in range(n):
        for j in range(i + 1):
            # Up move
            plt.plot([i, i + 1], 
                    [stock_tree[j, i], stock_tree[j, i + 1]], 
                    'g-', alpha=0.5, label='Up Move' if i==0 and j==0 else "")
            # Down move
            plt.plot([i, i + 1], 
                    [stock_tree[j, i], stock_tree[j + 1, i + 1]], 
                    'r-', alpha=0.5, label='Down Move' if i==0 and j==0 else "")
    
    plt.title(f"{title}\nTotal Nodes: {calculate_tree_size(n)}\nInitial Option Price: ${initial_option_price:.2f}")
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()

def binomial_option_pricer(S: float, K: float, T: float, r: float, sigma: float, 
                         n: int, option_type: str = 'call') -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Main function to price an option using the binomial model.
    """
    # Step 1: Calculate tree parameters
    u, d, p, dt = binomial_tree_parameters(S, K, T, r, sigma, n)
    
    # Step 2: Build stock price tree
    stock_tree = build_stock_price_tree(S, u, d, n)
    
    # Step 3: Calculate option payoff at expiration
    payoff = calculate_option_payoff(stock_tree, K, option_type)
    
    # Step 4: Perform backward induction
    option_tree = backward_induction(payoff, p, r, dt)
    
    return option_tree[0, 0], stock_tree, option_tree

def calculate_node_probabilities(n: int, p: float) -> np.ndarray:
    """
    Calculate the probability of reaching each node in the tree.
    
    Args:
        n: Number of time steps
        p: Probability of up move
    
    Returns:
        Array of probabilities for each node
    """
    probs = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            # Number of up moves = (i-j)
            # Number of down moves = j
            # Total moves = i
            probs[j, i] = binom.pmf(i-j, i, p)
    return probs

def build_stock_price_tree_custom(S: float, u: float, p: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the stock price tree with custom parameters and calculate probabilities.
    
    Args:
        S: Initial stock price
        u: Up factor
        p: Probability of up move
        n: Number of time steps
    
    Returns:
        Tuple of (stock price tree, probability tree)
    """
    d = 1/u  # Down factor
    tree = np.zeros((n + 1, n + 1))
    tree[0, 0] = S
    
    # Build price tree
    for i in range(1, n + 1):
        for j in range(i + 1):
            tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Calculate probabilities
    probs = calculate_node_probabilities(n, p)
    
    return tree, probs

def plot_custom_tree(stock_tree: np.ndarray, prob_tree: np.ndarray, 
                    title: str = "Custom Binomial Tree"):
    """
    Plot the binomial tree with probabilities.
    
    Args:
        stock_tree: Stock price tree
        prob_tree: Probability tree
        title: Plot title
    """
    n = stock_tree.shape[1] - 1
    plt.figure(figsize=(15, 10))
    
    # Plot stock prices and probabilities
    for i in range(n + 1):
        for j in range(i + 1):
            plt.scatter(i, stock_tree[j, i], color='blue', s=100)
            plt.text(i, stock_tree[j, i], 
                    f'S={stock_tree[j, i]:.0f}\nP={prob_tree[j, i]:.3f}',
                    ha='center', va='bottom')
    
    # Connect nodes
    for i in range(n):
        for j in range(i + 1):
            # Up move
            plt.plot([i, i + 1], 
                    [stock_tree[j, i], stock_tree[j, i + 1]], 
                    'g-', alpha=0.5, label='Up Move' if i==0 and j==0 else "")
            # Down move
            plt.plot([i, i + 1], 
                    [stock_tree[j, i], stock_tree[j + 1, i + 1]], 
                    'r-', alpha=0.5, label='Down Move' if i==0 and j==0 else "")
    
    plt.title(f"{title}\nInitial Price: ${stock_tree[0,0]:.0f}, Up Factor: {u}, Up Probability: {p}")
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculate option price using Black-Scholes formula.
    """
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def main():
    print("Welcome to the Binomial Options Pricer!")
    print("----------------------------------------")
    
    # Get user inputs
    #S, K, T, r, sigma, n, option_type = get_user_inputs()
    
    # Calculate Black-Scholes price for comparison
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    
    # Price the option
    option_price, stock_tree, option_tree = binomial_option_pricer(
        S, K, T, r, sigma, n, option_type
    )
    
    # Calculate intrinsic and time value
    if option_type == 'call':
        intrinsic_value = max(S - K, 0)
    else:  # put
        intrinsic_value = max(K - S, 0)
    time_value = option_price - intrinsic_value
    
    print(f"\nResults:")
    print(f"Current Stock Price: ${S:.2f}")
    print(f"Strike Price: ${K:.2f}")
    print(f"Time to Expiration: {T:.2f} years")
    print(f"Risk-free Rate: {r*100:.2f}%")
    print(f"Volatility: {sigma*100:.2f}%")
    print(f"Option Type: {option_type.capitalize()}")
    print(f"\nOption Value Breakdown:")
    print(f"Intrinsic Value: ${intrinsic_value:.4f}")
    print(f"Time Value: ${time_value:.4f}")
    print(f"Total Option Price: ${option_price:.4f}")
    print(f"Black-Scholes Price: ${bs_price:.4f}")
    print(f"Difference: ${option_price - bs_price:.4f}")
    
    # Show comparison with different strike prices
    if option_type == 'call':
        print("\nStrike Price Strategies (Call Options):")
        print("Strategy | Strike | Intrinsic | Time Value | Total Price | Break-even")
        print("------------------------------------------------------------------")
        
        # Deep ITM (20% below)
        strike_itm = S * 0.8
        price_itm, _, _ = binomial_option_pricer(S, strike_itm, T, r, sigma, n, option_type)
        intrinsic_itm = max(S - strike_itm, 0)
        time_val_itm = price_itm - intrinsic_itm
        breakeven_itm = strike_itm + price_itm
        print(f"Deep ITM  | ${strike_itm:.2f} | ${intrinsic_itm:.4f} | ${time_val_itm:.4f} | ${price_itm:.4f} | ${breakeven_itm:.2f}")
        
        # ATM
        strike_atm = S
        price_atm, _, _ = binomial_option_pricer(S, strike_atm, T, r, sigma, n, option_type)
        intrinsic_atm = max(S - strike_atm, 0)
        time_val_atm = price_atm - intrinsic_atm
        breakeven_atm = strike_atm + price_atm
        print(f"ATM      | ${strike_atm:.2f} | ${intrinsic_atm:.4f} | ${time_val_atm:.4f} | ${price_atm:.4f} | ${breakeven_atm:.2f}")
        
        # OTM (20% above)
        strike_otm = S * 1.2
        price_otm, _, _ = binomial_option_pricer(S, strike_otm, T, r, sigma, n, option_type)
        intrinsic_otm = max(S - strike_otm, 0)
        time_val_otm = price_otm - intrinsic_otm
        breakeven_otm = strike_otm + price_otm
        print(f"OTM      | ${strike_otm:.2f} | ${intrinsic_otm:.4f} | ${time_val_otm:.4f} | ${price_otm:.4f} | ${breakeven_otm:.2f}")
        
        print("\nStrategy Characteristics:")
        print("Deep ITM: High cost, high probability of profit, behaves like stock")
        print("ATM:     Moderate cost, balanced risk/reward, maximum leverage")
        print("OTM:     Low cost, needs big move to profit, highest risk/reward")
    
    # Plot the binomial tree
    plot_binomial_tree(stock_tree, option_tree, 
                      f"{option_type.capitalize()} Option Binomial Tree")

if __name__ == "__main__":
    main() 