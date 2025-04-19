# DeFi Strategy Generation API

This project provides a Python Flask API that generates a personalized DeFi investment strategy based on a user's profile and simulates its potential performance using Monte Carlo analysis. It leverages the Google Gemini API for strategy generation and DefiLlama for market data.

## Prerequisites

*   **pip** (Python package installer)
*   **Google Gemini API Key:** You need an API key from Google AI Studio ([https://aistudio.google.com/](https://aistudio.google.com/)).

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <repo-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables:**
    You need to set the `GEMINI_API_KEY` environment variable. Replace `'your_api_key_here'` with your actual key.
    ```bash
    export GEMINI_API_KEY='your_api_key_here'
    ```
    *Note: For production environments, use a more secure method for managing secrets (e.g., `.env` files with `python-dotenv`, system environment variables, or secrets management services).*

## Running the API

Start the Flask development server:

```bash
python api.py
```

By default, the server runs on port 5000 (`http://0.0.0.0:5000/`). If port 5000 is already in use, you can specify a different port using the `PORT` environment variable:

```bash
PORT=5001 python api.py
```

The API will then be accessible at `http://0.0.0.0:5001/`.

## API Endpoint

### `POST /strategy`

Generates a DeFi strategy and runs a portfolio simulation.

*   **Method:** `POST`
*   **Headers:**
    *   `Content-Type: application/json`
*   **Request Body:** A JSON object representing the user's profile.

    ```json
    {
      "fund": 50000,
      "max_exposure": 0.9,
      "risk_level": "medium",        // "low", "medium", or "high"
      "time_horizon": 12,          // Integer, months
      "kill_switch": -0.1,         // Float, e.g., -0.1 for -10%
      "investment_goals": ["capital_appreciation", "passive_income"], // Array of strings
      "preferred_activities": ["lending", "staking"] // Array of strings (e.g., "lending", "staking", "yield_farming", "trading")
    }
    ```

*   **Success Response (200 OK):** A JSON object containing the generated strategy and simulation summary.

    ```json
    {
      "strategy": {
        "trend_ma_fast": 12, // Example value
        "trend_ma_slow": 48, // Example value
        "allocations": [
          { "protocol": "Aave", "weight_pct": 0.45 }, // Example value
          { "protocol": "Lido", "weight_pct": 0.45 }  // Example value
        ]
      },
      "simulation_summary": {
        "mean_final_value": 58234.56, // Example value
        "median_final_value": 57100.12,
        "std_dev_final_value": 8450.78,
        "percentile_5th": 45300.90,
        "percentile_95th": 72500.60,
        "num_simulations": 1000,
        // Could include an "error" key if only simulation failed
        // e.g., "error": "Failed to fetch historical return data"
      }
      // simulation_summary might be null if prerequisite data was missing
    }
    ```

*   **Error Response (4xx or 5xx):** A JSON object describing the error.

    ```json
    // Example: Bad Request (400)
    {
      "error": "Request must be JSON"
    }

    // Example: Server Error (500)
    {
      "error": "Failed to fetch market data"
    }

    // Example: AI Model Error (500)
    {
        "error": "Failed to decode strategy from AI model",
        "details": "Expecting value: line 1 column 1 (char 0)"
    }
    ```

## Usage with a Next.js App

The recommended way to use this Python API from a Next.js application is to call it from one of your Next.js **API routes** (server-side). This avoids exposing your Python API directly to the browser and simplifies CORS (Cross-Origin Resource Sharing) configuration.

1.  **Ensure API is Running:** Make sure your Python Flask API server is running (e.g., `PORT=5001 python api.py`).

2.  **Create a Next.js API Route:** Create a file like `pages/api/generate-strategy.js` (or `.ts`).

3.  **Call the Python API from Next.js:** Use `fetch` within the Next.js API route handler to make the POST request to your Python API.

    ```javascript
    // pages/api/generate-strategy.js
    import type { NextApiRequest, NextApiResponse } from 'next';

    // Define the expected input structure from your frontend if needed
    // type UserProfileInput = { ... };

    // Define the expected output structure from the Python API
    // type StrategyApiResponse = { strategy: ..., simulation_summary: ... };

    export default async function handler(
      req: NextApiRequest,
      res: NextApiResponse // Adjust type based on StrategyApiResponse | { error: string }
    ) {
      if (req.method !== 'POST') {
        res.setHeader('Allow', ['POST']);
        return res.status(405).end(`Method ${req.method} Not Allowed`);
      }

      try {
        const userProfile = req.body; // Get user profile from frontend request

        // Get the Python API URL from environment variables
        const pythonApiUrl = process.env.PYTHON_STRATEGY_API_URL || 'http://localhost:5001/strategy';

        const apiResponse = await fetch(pythonApiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(userProfile),
        });

        // Check if the Python API call was successful
        if (!apiResponse.ok) {
          let errorBody = 'Python API request failed';
          try {
            const errorJson = await apiResponse.json();
            errorBody = errorJson.error || JSON.stringify(errorJson);
          } catch (e) {
            // Could not parse error body as JSON
            errorBody = await apiResponse.text();
          }
          console.error(`Python API Error (${apiResponse.status}): ${errorBody}`);
          return res.status(apiResponse.status).json({ error: `Strategy generation failed: ${errorBody}` });
        }

        // Parse the successful response from the Python API
        const data = await apiResponse.json();

        // Send the combined result back to the Next.js frontend
        res.status(200).json(data);

      } catch (error) {
        console.error('Error in Next.js API route:', error);
        res.status(500).json({ error: 'Internal Server Error in Next.js route' });
      }
    }
    ```

4.  **Set Environment Variable in Next.js:** Add the Python API URL to your Next.js environment variables (e.g., in `.env.local`):
    ```
    PYTHON_STRATEGY_API_URL=http://localhost:5001/strategy
    ```
    *Remember to configure the appropriate URL for your production environment.* Prefixing with `NEXT_PUBLIC_` is **not** needed here, as this variable is only used server-side in the API route.

5.  **Call the Next.js API Route from Frontend:** Your Next.js frontend components will now call `/api/generate-strategy` (your Next.js route), not the Python API directly.

    ```javascript
    // Example frontend component function
    async function generateStrategy(userProfileData) {
      try {
        const response = await fetch('/api/generate-strategy', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(userProfileData),
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(result.error || 'Failed to generate strategy');
        }

        console.log('Strategy and Simulation:', result);
        // Update your UI state with the result
        return result;

      } catch (error) {
        console.error('Error fetching strategy:', error);
        // Handle error in the UI
      }
    }
    ```

## Deployment Considerations

*   You need to deploy both the Python Flask API and your Next.js application.
*   Ensure the deployed Next.js application (specifically its server-side API route) can network-reach the deployed Python Flask API's URL.
*   Use a production-grade WSGI server (like Gunicorn or Waitress) instead of the Flask development server for the Python API in production.
*   Manage environment variables and secrets securely in your deployment environment. 