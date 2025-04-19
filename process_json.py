import os
import json
import google.generativeai as genai
from google.api_core import exceptions
import requests
from market_data import get_filtered_protocols # Import the function
import pandas as pd # Added for potential data handling if needed here
import numpy as np  # Added for potential array handling if needed here
from portfolio_simulation import ( # Added import
    fetch_historical_returns,
    estimate_parameters,
    run_monte_carlo,
    summarize_results
)

# Configure the API key
try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)
except Exception as e:
    print(f"An error occurred during API configuration: {e}")
    exit(1)

# Function to read file content
def read_file_content(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

# Corrected variable name
prompt_template = """
You are an AI powered DeFi assistant.

Your task is to take this user's investment profile and suggest ONE optimal, rule based investment strategy that:
• Trades only currently active DeFi protocols/tokens provided in the Market Data (which are already filtered by user's preferred activities).
• Respects the user's risk limits.
• Keeps the user's exposure within the max_exposure limit.
• Aligns with the user's investment goals.

--- User Profile ---
{user_profile_json}

The user profile contains the following keys:
- "fund" (float, dollars)
- "max_exposure" (float, 0-1)
- "risk_level" (string, "low", "medium", "high")
- "time_horizon" (integer, months)
- "kill_switch" (float, e.g. -0.05 for -5%)
- "investment_goals" (array of strings, e.g., ["passive_income", "capital_appreciation"])
- "preferred_activities" (array of strings, e.g., ["lending", "staking"]) - Note: Market Data is already filtered based on these.

--- Market Data ---
{protocol_data_json}

════════ MOVING-AVERAGE RULES ════════
Base on the user's **time-horizon**, then tweak by **risk level**.

1. Pick the BASE pair from horizon alone
   • 3-6 months  ->  fast 5-10 d,   slow 20-30 d  
   • 6-12 months ->  fast 10-15 d,  slow 35-60 d  
   • 12-24 months -> fast 20-30 d,  slow 80-120 d  

2. Adjust for risk tolerance
   • Low-risk  ->  lengthen both MAs ~ +20 % (smoother, fewer trades)  
   • Medium    ->  keep the base values  
   • High-risk ->  shorten both MAs ~ -20 % (quicker in/out)

3. Make sure fast < slow - if an adjustment flips them, pull the fast back under the slow.

════════ ASSET-SELECTION RULES ════════
1. Start with the protocols provided in the Market Data (already filtered by `preferred_activities`).
2. Select protocols that align well with the user's `investment_goals`.
3. From the aligned protocols, rank the remainder by **TVL (highest first)**.
4. Pick the top **<= 3** names so the strategy is simple to follow.
5. Allocate weights across the chosen names, but cap total exposure at `max_exposure`.

════════ REQUIRED OUTPUT KEYS ════════
- "trend_ma_fast" (integer, days)  
- "trend_ma_slow" (integer, days)  
- "allocations" (array of {{ "protocol": <name>, "weight_pct": 0-1 }})

════════ OUTPUT FORMAT ════════
Respond with **exactly** one JSON array containing one object, no commentary:

[
  {{
    "trend_ma_fast": 10,
    "trend_ma_slow": 40,
    "allocations": [
      {{ "protocol": "Aave", "weight_pct": 0.225 }},
      {{ "protocol": "Uniswap", "weight_pct": 0.225 }}
    ]
  }}
]
"""

# --- Data Loading and Preparation ---

# Read the user profile JSON data
json_file = "test.json"
user_profile_str = read_file_content(json_file)
if user_profile_str is None:
    exit(1)

# Parse the user profile JSON string into a Python object
try:
    user_profile = json.loads(user_profile_str)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {json_file}: {e}")
    exit(1)

# Fetch market data
print("Fetching market data...")
try:
    # Define preferred activities (example, adjust as needed)
    preferred_activities = ["lending", "staking"]
    protocol_data = get_filtered_protocols(preferred_activities=preferred_activities)
    print(f"Fetched {len(protocol_data)} protocols.")
except requests.RequestException as e:
    print(f"Error fetching market data from API: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while fetching market data: {e}")
    exit(1)

# Convert Python objects back to formatted JSON strings for the prompt
user_profile_json_str = json.dumps(user_profile, indent=2)
protocol_data_json_str = json.dumps(protocol_data, indent=2)

# Format the final prompt with the actual data
llm_input = prompt_template.format(
    user_profile_json=user_profile_json_str,
    protocol_data_json=protocol_data_json_str
)

# Initialize the Generative Model
try:
    model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') # Using gemini-2.5-pro-exp-03-25 as a default
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit(1)

# Send to Gemini and get the response
try:
    print("Sending request to Gemini...")
    response = model.generate_content(llm_input)
    print("--- Gemini Response ---")
    # Handle potential lack of 'text' attribute gracefully
    if hasattr(response, 'text'):
        print(response.text)
    elif hasattr(response, 'parts'):
         # Handle cases where the response might be structured differently
        full_response = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        print(full_response)
    else:
        print("Could not extract text from response.")
        print(f"Full response object: {response}")
    print("-----------------------")

except exceptions.InvalidArgument as e:
     print(f"Error: Invalid argument provided to the API. Check your prompt or input data. Details: {e}")
except exceptions.PermissionDenied as e:
     print(f"Error: Permission denied. Check your API key and its permissions. Details: {e}")
except exceptions.ResourceExhausted as e:
     print(f"Error: API quota exceeded. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred while calling the Gemini API: {e}")
    exit(1)

# --- Portfolio Simulation ---
# Extract necessary info after getting LLM response
strategy_json_str = None
if hasattr(response, 'text'):
    strategy_json_str = response.text
elif hasattr(response, 'parts'):
    strategy_json_str = "".join(part.text for part in response.parts if hasattr(part, 'text'))

if strategy_json_str:
    try:
        # Attempt to parse the JSON response from LLM
        # Need to handle potential markdown ```json ... ``` wrapping
        if strategy_json_str.strip().startswith("```json"):
            strategy_json_str = strategy_json_str.strip()[7:-3].strip()
        elif strategy_json_str.strip().startswith("```"):
             strategy_json_str = strategy_json_str.strip()[3:-3].strip()

        strategy_data = json.loads(strategy_json_str)

        # Assuming the response is a list containing one strategy object
        if isinstance(strategy_data, list) and len(strategy_data) > 0:
            strategy = strategy_data[0]
            allocations = strategy.get("allocations")
            initial_capital = user_profile.get("fund")
            time_horizon_months = user_profile.get("time_horizon")

            if allocations and initial_capital is not None and time_horizon_months is not None:
                print("\n--- Running Portfolio Simulation ---")
                protocol_names = [a['protocol'] for a in allocations]

                # 1. Fetch Historical Data (using mock function for now)
                # Consider adding a lookback period relevant to time_horizon?
                returns_df = fetch_historical_returns(protocol_names)

                if returns_df is not None:
                    # 2. Estimate Parameters
                    mean_returns, cov_matrix = estimate_parameters(returns_df)

                    if mean_returns is not None and cov_matrix is not None:
                        # 3. Run Monte Carlo Simulation
                        final_portfolio_values = run_monte_carlo(
                            initial_capital=initial_capital,
                            time_horizon_months=time_horizon_months,
                            allocations=allocations,
                            mean_returns=mean_returns,
                            cov_matrix=cov_matrix,
                            num_simulations=1000 # Example simulation count
                        )

                        if final_portfolio_values is not None:
                            # 4. Summarize Results
                            simulation_summary = summarize_results(final_portfolio_values)

                            if simulation_summary:
                                print("\n--- Simulation Results ---")
                                print(json.dumps(simulation_summary, indent=2))
                                print("--------------------------")
                            else:
                                print("Failed to summarize simulation results.")
                        else:
                             print("Monte Carlo simulation failed to produce results.")
                    else:
                        print("Failed to estimate statistical parameters from historical data.")
                else:
                    print("Failed to fetch or generate historical return data for simulation.")
            else:
                print("Could not extract necessary 'allocations', 'fund', or 'time_horizon' for simulation.")
        else:
            print("Could not parse valid strategy object from LLM response.")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Raw response string was: {strategy_json_str}")
    except Exception as e:
        print(f"An unexpected error occurred during portfolio simulation: {e}")
else:
    print("No valid response text received from Gemini to run simulation.")