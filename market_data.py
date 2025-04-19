import requests
import backoff
import json

DEFI_LLAMA_API = "https://api.llama.fi/protocols"

ACTIVITY_CATEGORY_MAP = {
    "lending": "lending",
    "staking": "staking",
    "yield_farming": "yield",
    "liquidity_providing": "dexes",
    "trading": "dexes",
}

# Default list of main chains to consider
DEFAULT_MAIN_CHAINS = [
    "ethereum",
    "polygon",
    "arbitrum",
    "solana",
]

@backoff.on_exception(backoff.expo,
                      (requests.RequestException, ValueError),
                      max_tries=3)
def get_filtered_protocols(
    preferred_activities=None,
    min_tvl=1_000_000,
    target_chains=None # Added parameter for target chains
):
    """Return a trimmed list of live protocols on specified chains."""

    if target_chains is None:
        target_chains = DEFAULT_MAIN_CHAINS # Use default if none provided
    target_chains_lower = {chain.lower() for chain in target_chains}

    r = requests.get(DEFI_LLAMA_API, timeout=10)
    r.raise_for_status()
    protocols = r.json()

    allowed_categories = (
        set(ACTIVITY_CATEGORY_MAP.values()) if not preferred_activities
        else {ACTIVITY_CATEGORY_MAP[a.lower()]
              for a in preferred_activities
              if a.lower() in ACTIVITY_CATEGORY_MAP}
    )

    out = []
    for p in protocols:
        protocol_chains_lower = {chain.lower() for chain in p.get("chains", [])}
        tvl = p.get("tvl") # Get the TVL value, could be None
        symbol = p.get("symbol") # Get symbol, could be None
        if (
            tvl is not None and tvl > min_tvl and
            any(pc in target_chains_lower for pc in protocol_chains_lower) and
            p.get("category", "").lower() in allowed_categories and
            symbol is not None and symbol != "N/A" # Check for valid symbol
        ):
            out.append({
                "name": p["name"],
                "category": p["category"],
                "tvl": p["tvl"],
                "chains": p["chains"],
                "symbol": p.get("symbol", "N/A")
            })
    return out
