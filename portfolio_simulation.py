import pandas as pd
import numpy as np

def fetch_historical_returns(protocols: list[str], lookback_days: int = 365) -> pd.DataFrame | None:
    """
    Placeholder function to fetch historical daily returns for given protocols.
    Replace this with actual data fetching logic (e.g., from DefiLlama).

    Args:
        protocols: List of protocol names.
        lookback_days: Number of days of historical data to fetch.

    Returns:
        A pandas DataFrame with dates as index and protocol returns as columns,
        or None if data fetching fails.
    """
    print(f"--- MOCK DATA ---: Simulating fetching data for {protocols} over {lookback_days} days.")
    # In a real scenario, you would query an API (like DefiLlama) here.
    # Returning mock data for demonstration:
    if not protocols:
        return None
    dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback_days, freq='D')
    data = {protocol: np.random.normal(loc=0.0005, scale=0.02, size=lookback_days) for protocol in protocols}
    mock_returns_df = pd.DataFrame(data, index=dates)
    print(f"--- MOCK DATA ---: Generated mock DataFrame with shape {mock_returns_df.shape}")
    return mock_returns_df

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