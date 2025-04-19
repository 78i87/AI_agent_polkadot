import os
import json
import requests
import google.generativeai as genai
from google.api_core import exceptions
from flask import Flask, request, jsonify

from market_data import get_filtered_protocols
from portfolio_simulation import (
    fetch_historical_returns,
    estimate_parameters,
    run_monte_carlo,
    summarize_results
)

# --- Configuration ---
try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    MODEL = genai.GenerativeModel('gemini-2.5-pro-preview-03-25') # Initialize model once
except KeyError:
    print("FATAL ERROR: GEMINI_API_KEY environment variable not set.")
    exit(1) # Exit if API key is missing
except Exception as e:
    print(f"FATAL ERROR: An error occurred during API configuration or model initialization: {e}")
    exit(1) # Exit on other config errors


# --- Prompt Template ---
# (Keep the same prompt template as in process_json.py)
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

# --- Helper Function to Call Gemini ---
def get_gemini_strategy(user_profile_json: str, protocol_data_json: str) -> str | None:
    """Formats the prompt, calls the Gemini API, and returns the text response."""
    llm_input = prompt_template.format(
        user_profile_json=user_profile_json,
        protocol_data_json=protocol_data_json
    )
    try:
        response = MODEL.generate_content(llm_input)
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts'):
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            print(f"Warning: Could not extract text from Gemini response. Full response: {response}")
            return None
    except exceptions.InvalidArgument as e:
         print(f"Error: Invalid argument provided to Gemini API. Details: {e}")
         raise # Re-raise specific exceptions to be caught by the endpoint
    except exceptions.PermissionDenied as e:
         print(f"Error: Permission denied for Gemini API. Check API key. Details: {e}")
         raise
    except exceptions.ResourceExhausted as e:
         print(f"Error: Gemini API quota exceeded. Details: {e}")
         raise
    except Exception as e:
        print(f"An unexpected error occurred while calling the Gemini API: {e}")
        raise # Re-raise general exceptions


# --- Helper Function for Portfolio Simulation ---
def run_portfolio_simulation(strategy: dict, user_profile: dict) -> dict | None:
    """Runs the portfolio simulation based on strategy and user profile."""
    allocations = strategy.get("allocations")
    initial_capital = user_profile.get("fund")
    time_horizon_months = user_profile.get("time_horizon")

    if not allocations or initial_capital is None or time_horizon_months is None:
        print("Warning: Missing 'allocations', 'fund', or 'time_horizon' for simulation.")
        return None

    try:
        print("--- Running Portfolio Simulation ---")
        protocol_names = [a['protocol'] for a in allocations]
        returns_df = fetch_historical_returns(protocol_names)

        if returns_df is None:
            print("Error: Failed to fetch historical return data for simulation.")
            return {"error": "Failed to fetch historical return data"}

        mean_returns, cov_matrix = estimate_parameters(returns_df)

        if mean_returns is None or cov_matrix is None:
            print("Error: Failed to estimate statistical parameters from historical data.")
            return {"error": "Failed to estimate parameters from historical data"}

        # Get risk_level from the user profile
        risk_level = user_profile.get("risk_level", "medium") # Default to medium if missing

        final_portfolio_values = run_monte_carlo(
            initial_capital=initial_capital,
            time_horizon_months=time_horizon_months,
            allocations=allocations,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_level=risk_level, # Pass the risk level
            num_simulations=1000 # Keep consistent or make configurable?
        )

        if final_portfolio_values is None:
            print("Error: Monte Carlo simulation failed to produce results.")
            return {"error": "Monte Carlo simulation failed"}

        simulation_summary = summarize_results(final_portfolio_values)

        if simulation_summary:
            print("--- Simulation Complete ---")
            return simulation_summary
        else:
            print("Error: Failed to summarize simulation results.")
            return {"error": "Failed to summarize simulation results"}

    except Exception as e:
        print(f"An unexpected error occurred during portfolio simulation: {e}")
        return {"error": f"Unexpected simulation error: {str(e)}"}


# --- Flask App ---
app = Flask(__name__)

@app.route('/strategy', methods=['POST'])
def generate_strategy_endpoint():
    """
    API endpoint to generate strategy and run simulation.
    Expects JSON body with user_profile.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    user_profile = request.get_json()
    if not user_profile:
        return jsonify({"error": "Missing user profile data in request body"}), 400

    print(f"Received request for user profile: {json.dumps(user_profile, indent=2)}")

    # --- 1. Fetch Market Data ---
    try:
        preferred_activities = user_profile.get("preferred_activities", ["lending", "staking"]) # Default if not provided
        print(f"Fetching market data for activities: {preferred_activities}...")
        protocol_data = get_filtered_protocols(preferred_activities=preferred_activities)
        if not protocol_data:
             print("Warning: No protocols found matching preferred activities.")
             # Proceed anyway, let Gemini handle it or return empty strategy?
             # For now, proceed.
        print(f"Fetched {len(protocol_data)} protocols.")
    except requests.RequestException as e:
        print(f"Error fetching market data: {e}")
        return jsonify({"error": "Failed to fetch market data"}), 500
    except Exception as e:
        print(f"Unexpected error fetching market data: {e}")
        return jsonify({"error": "Unexpected error fetching market data"}), 500

    # --- 2. Call Gemini for Strategy ---
    try:
        user_profile_json_str = json.dumps(user_profile, indent=2)
        protocol_data_json_str = json.dumps(protocol_data, indent=2)

        print("Sending request to Gemini...")
        strategy_json_str = get_gemini_strategy(user_profile_json_str, protocol_data_json_str)

        if not strategy_json_str:
            print("Error: Received no text response from Gemini.")
            return jsonify({"error": "Failed to get strategy from AI model (no response)"}), 500

        print(f"""--- Raw Gemini Response --- 
{strategy_json_str}
-------------------------""")

        # Attempt to parse the JSON response from LLM
        # Handle potential markdown ```json ... ``` wrapping
        if strategy_json_str.strip().startswith("```json"):
            strategy_json_str = strategy_json_str.strip()[7:-3].strip()
        elif strategy_json_str.strip().startswith("```"):
            strategy_json_str = strategy_json_str.strip()[3:-3].strip()

        strategy_data = json.loads(strategy_json_str)

        # Assuming the response is a list containing one strategy object
        if not isinstance(strategy_data, list) or len(strategy_data) == 0:
             print("Error: Gemini response was not a list or was empty.")
             return jsonify({"error": "AI model returned unexpected format"}), 500

        strategy = strategy_data[0] # Get the actual strategy object

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Raw response was: {strategy_json_str}")
        return jsonify({"error": "Failed to decode strategy from AI model", "details": str(e)}), 500
    except (exceptions.InvalidArgument, exceptions.PermissionDenied, exceptions.ResourceExhausted) as e:
         # Specific Gemini API errors caught by helper
         return jsonify({"error": "AI model API error", "details": str(e)}), 500
    except Exception as e:
        print(f"An unexpected error occurred during Gemini interaction: {e}")
        return jsonify({"error": "Unexpected error generating strategy", "details": str(e)}), 500


    # --- 3. Run Portfolio Simulation ---
    simulation_results = run_portfolio_simulation(strategy, user_profile)


    # --- 4. Return Combined Results ---
    final_response = {
        "strategy": strategy,
        "simulation_summary": simulation_results # Will be None or contain error if failed
    }
    print(f"""--- Final Response --- 
{json.dumps(final_response, indent=2)}
--------------------""")
    return jsonify(final_response), 200


# --- Main Execution ---
if __name__ == '__main__':
    # Make sure to install Flask: pip install Flask
    # Consider using waitress or gunicorn for production deployment
    # Host='0.0.0.0' makes it accessible on the network
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False) # Turn debug off for production/security 