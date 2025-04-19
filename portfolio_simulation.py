import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Union, Optional, Dict, List, Tuple

def fetch_historical_returns(protocols: list[str], lookback_days: int = 365) -> pd.DataFrame | None:
    """
    Fetches historical daily returns for given DeFi protocols using DefiLlama API.
    
    This function can fetch different metrics from DefiLlama and calculate returns:
    - Total Value Locked (TVL) changes (default)
    - Price changes for protocol tokens
    - Revenue and fees

    Args:
        protocols: List of protocol names/slugs as used in DefiLlama (e.g., "aave", "uniswap").
        lookback_days: Number of days of historical data to fetch.

    Returns:
        A pandas DataFrame with dates as index and protocol returns as columns,
        or None if data fetching fails for all protocols.
    """
    if not protocols:
        print("No protocols specified.")
        return None
        
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_days)
    print(f"Fetching data from {start_date} to {end_date} for: {', '.join(protocols)}")
    
    # Container for all protocol returns
    all_returns = {}
    
    for protocol in protocols:
        try:
            # First try to get TVL-based returns (most reliable)
            df = _fetch_tvl_based_returns(protocol, start_date, end_date)
            
            if df is None or df.empty:
                # Try token-based returns as fallback
                df = _fetch_token_based_returns(protocol, start_date, end_date)
            
            if df is None or df.empty:
                print(f"No data found for {protocol}, skipping...")
                continue
                
            # Add to our collection
            all_returns[protocol] = df
            print(f"Successfully fetched returns for {protocol} ({len(df)} days)")
            
        except Exception as e:
            print(f"Error processing {protocol}: {str(e)}")
    
    if not all_returns:
        print("Failed to fetch data for any protocols")
        return None
    
    # Combine all protocol returns into a single DataFrame
    # First convert all series to dataframes with protocol as column name
    dfs = []
    for protocol, data in all_returns.items():
        if isinstance(data, pd.Series):
            # Ensure the index is a datetime and contains no duplicates
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # If there are duplicate dates, take the last value for each date
            if data.index.duplicated().any():
                data = data.groupby(data.index).last()
                
            df = data.to_frame(name=protocol)
            dfs.append(df)
        else:
            # Already a DataFrame
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
                
            # If there are duplicate dates, take the last value for each date
            if data.index.duplicated().any():
                data = data.groupby(data.index).last()
                
            data.columns = [protocol]
            dfs.append(data)
    
    # Merge all dataframes on date
    if dfs:
        returns_df = pd.concat(dfs, axis=1)
        
        # Convert string dates to datetime if needed
        if not isinstance(returns_df.index, pd.DatetimeIndex):
            returns_df.index = pd.to_datetime(returns_df.index)
            
        # Create a full date range with DatetimeIndex
        date_range = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D')
        
        # Reindex using the proper date range
        returns_df = returns_df.reindex(date_range)
        
        # Forward fill for up to 3 days to handle missing data
        returns_df = returns_df.fillna(method='ffill', limit=3)
        
        # Fill remaining NAs with 0 (no change)
        returns_df = returns_df.fillna(0)
        
        print(f"Final returns DataFrame shape: {returns_df.shape}")
        return returns_df
    
    return None

def _fetch_tvl_based_returns(protocol: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.Series]:
    """
    Fetch protocol TVL data and compute daily returns based on TVL changes.
    """
    try:
        # DeFiLlama protocol endpoint
        url = f"https://api.llama.fi/protocol/{protocol}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching TVL data for {protocol}: HTTP {response.status_code}")
            return None
            
        data = response.json()
        
        # Extract TVL time series
        if 'tvl' not in data:
            print(f"No TVL data found for {protocol}")
            return None
            
        # Convert timestamp to date and filter by our date range
        tvl_series = []
        for entry in data['tvl']:
            timestamp = entry['date']
            date = datetime.fromtimestamp(timestamp).date()
            if start_date <= date <= end_date:
                tvl_series.append({'date': date, 'tvl': entry['totalLiquidityUSD']})
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(tvl_series)
        if df.empty:
            print(f"No TVL data in date range for {protocol}")
            return None
            
        df.sort_values('date', inplace=True)
        
        # Check for duplicate dates and keep only the latest entry for each date
        if df.duplicated('date').any():
            df = df.drop_duplicates('date', keep='last')
            
        df.set_index('date', inplace=True)
        
        # Calculate daily returns
        returns = df['tvl'].pct_change()
        
        # Drop first row (NaN return)
        returns = returns.dropna()
        
        return returns
        
    except Exception as e:
        print(f"Error calculating TVL-based returns for {protocol}: {e}")
        return None

def _fetch_token_based_returns(protocol: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.Series]:
    """
    Fetch protocol token price data and compute daily returns based on price changes.
    """
    try:
        # First get protocol info to find associated token
        url = f"https://api.llama.fi/protocol/{protocol}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching protocol data for {protocol}: HTTP {response.status_code}")
            return None
            
        data = response.json()
        
        # Check if protocol has a token
        if not data.get('gecko_id') and not data.get('symbol'):
            print(f"No token found for {protocol}")
            return None
            
        token_id = data.get('gecko_id', data.get('symbol', '').lower())
        
        # Get coin historical price data
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())
        
        url = f"https://coins.llama.fi/chart/{token_id}?start={start_timestamp}&span=1d"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching price data for {protocol} token: HTTP {response.status_code}")
            return None
            
        price_data = response.json()
        
        if 'coins' not in price_data or token_id not in price_data['coins']:
            print(f"No price data found for {protocol} token")
            return None
            
        # Extract price series
        prices = price_data['coins'][token_id]['prices']
        
        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        
        # Check for duplicate dates and keep only the latest entry for each date
        if df.duplicated('date').any():
            df = df.drop_duplicates('date', keep='last')
            
        df.set_index('date', inplace=True)
        
        # Calculate daily returns
        returns = df['price'].pct_change()
        
        # Drop first row (NaN return)
        returns = returns.dropna()
        
        return returns
        
    except Exception as e:
        print(f"Error calculating token-based returns for {protocol}: {e}")
        return None
        
def _get_protocol_info(protocol: str) -> Dict:
    """Helper function to get protocol information from DefiLlama."""
    url = f"https://api.llama.fi/protocol/{protocol}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to get protocol info: HTTP {response.status_code}")
        
    return response.json()

def estimate_parameters(returns_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimates the mean daily return vector (mu) and the covariance matrix (Sigma).

    Args:
        returns_df: DataFrame of historical daily returns.

    Returns:
        A tuple containing (mean_returns, covariance_matrix), or (None, None) if estimation fails.
    """
    if returns_df is None or returns_df.empty:
        return None, None
    try:
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        return mean_returns, cov_matrix
    except Exception as e:
        print(f"Error estimating parameters: {e}")
        return None, None

def run_monte_carlo(
    initial_capital: float,
    time_horizon_months: int,
    allocations: list[dict],
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_simulations: int = 1000,
    # rebalance_frequency_days: int = 30 # Rebalancing logic can be added later
) -> np.ndarray | None:
    """
    Runs a Monte Carlo simulation for portfolio value projection.

    Args:
        initial_capital: Starting investment amount.
        time_horizon_months: Investment duration in months.
        allocations: List of dicts [{"protocol": name, "weight_pct": weight}].
        mean_returns: Expected daily mean return vector for assets.
        cov_matrix: Covariance matrix of daily returns for assets.
        num_simulations: Number of simulation paths to generate.

    Returns:
        An array of final portfolio values for each simulation, or None if simulation fails.
    """
    if mean_returns is None or cov_matrix is None or not allocations:
        print("Error: Missing mean returns, covariance matrix, or allocations for simulation.")
        return None

    try:
        num_assets = len(allocations)
        if num_assets != len(mean_returns) or num_assets != cov_matrix.shape[0]:
             print("Error: Mismatch between allocations and return/covariance data.")
             return None

        weights = np.array([a['weight_pct'] for a in allocations])
        # Ensure weights sum <= 1 (respecting max_exposure implicitly handled by LLM allocation)
        if weights.sum() > 1.001: # Allow for small floating point inaccuracies
             print(f"Warning: Allocation weights sum to {weights.sum()}, should be <= 1.")
             # Optional: Normalize weights if they exceed 1, or just proceed if slightly over
             # weights = weights / weights.sum()

        num_days = time_horizon_months * 30 # Approximate days
        all_final_values = np.zeros(num_simulations)

        for i in range(num_simulations):
            portfolio_value = initial_capital
            # Simulate day by day - simple compounding without rebalancing for now
            daily_returns_sim = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
            portfolio_daily_returns = np.dot(daily_returns_sim, weights)

            # Calculate compounded growth
            cumulative_returns = np.cumprod(1 + portfolio_daily_returns)
            final_value = portfolio_value * cumulative_returns[-1]
            all_final_values[i] = final_value

        return all_final_values

    except Exception as e:
        print(f"Error during Monte Carlo simulation: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None


def summarize_results(final_values: np.ndarray) -> dict | None:
    """
    Calculates summary statistics from the simulation results.

    Args:
        final_values: Array of final portfolio values from simulations.

    Returns:
        A dictionary containing summary statistics, or None if input is invalid.
    """
    if final_values is None or len(final_values) == 0:
        return None
    try:
        summary = {
            "mean_final_value": np.mean(final_values),
            "median_final_value": np.median(final_values),
            "std_dev_final_value": np.std(final_values),
            "percentile_5th": np.percentile(final_values, 5),
            "percentile_95th": np.percentile(final_values, 95),
            "num_simulations": len(final_values)
        }
        return summary
    except Exception as e:
        print(f"Error summarizing results: {e}")
        return None 