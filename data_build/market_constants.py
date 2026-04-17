"""
Two categories:
  1. Keyword/title mappings  — which real-world terms map to each vertical
                               and campaign type. Update when better proxies
                               are found.
  2. Tuning constants        — window sizes, thresholds, retry counts.
                               Update when environment dynamics change.

VERTICAL_BENCHMARKS are loaded from vertical_benchmarks.yaml at runtime
via _load_vertical_benchmarks() in market_data_provider.py — not here.
"""

from typing import Dict, List
from pathlib import Path
import yaml

VERTICAL_KEYWORDS: Dict[str, Dict[str, str]] = {
    "ecommerce": {
        "branded": "nike",
        "generic": "running shoes",
        "display": "online shopping",
        "fred":    "RSXFS",       # Retail sales ex food services
    },
    "saas": {
        "branded": "salesforce",
        "generic": "crm software",
        "display": "business software",
        "fred":    "DGORDER",     # Durable goods orders — tech proxy
    },
    "travel": {
        "branded": "expedia",
        "generic": "cheap flights",
        "display": "travel deals",
        "fred":    "TOTALSA",     # Total vehicle sales — travel proxy
    },
    "finance": {
        "branded": "chase bank",
        "generic": "personal loan",
        "display": "financial services",
        "fred":    "UMCSENT",     # Consumer sentiment
    },
}

WIKIPEDIA_TITLES: Dict[str, List[str]] = {
    "ecommerce": [
        "Nike,_Inc.",
        "Athletic_shoes",
        "Online_shopping",
    ],
    "saas": [
        "Salesforce",
        "Customer_relationship_management",
        "Software_as_a_service",
    ],
    "travel": [
        "Expedia_Group",
        "Low-cost_carrier",
        "Tourism",
    ],
    "finance": [
        "JPMorgan_Chase",
        "Personal_loan",
        "Financial_services",
    ],
}

# ---------------------------------------------------------------------------
# Window and dataset sizing
# ---------------------------------------------------------------------------

# Number of days in one episode window
EPISODE_WINDOW_DAYS: int = 30

# Total lookback days to fetch from data sources
# Must be >= EPISODE_WINDOW_DAYS
DATASET_LOOKBACK_DAYS: int = 90

# Days to slide window forward per episode (Mode 2 — sequential)
# 7 = one week slide per episode, preserving temporal autocorrelation
SLIDE_STEP_DAYS: int = 7

# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

SPIKE_THRESHOLD: float = 1.5

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

# Maximum age of cached data in days before it is treated as stale
# and a live re-fetch is attempted
CACHE_EXPIRY_DAYS: int = 7

# ---------------------------------------------------------------------------
# Wikipedia fetch reliability
# ---------------------------------------------------------------------------
WIKIPEDIA_MAX_RETRIES: int = 3



def _load_vertical_benchmarks() -> Dict:
    """
    Load vertical benchmark data from vertical_benchmarks.yaml.
    This file contains industry benchmark CVRs, CPCs, volatility,
    and seasonal multipliers per vertical — updated quarterly.

    Raises FileNotFoundError if the file is missing — it is required.
    """
    benchmarks_file = Path(__file__).parent / "vertical_benchmarks.yaml"
    if not benchmarks_file.exists():
        raise FileNotFoundError(
            f"vertical_benchmarks.yaml not found at {benchmarks_file}. "
            f"This file is required for MarketDataProvider. "
            f"See the project README for the expected format."
        )
    with open(benchmarks_file) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        raise ValueError(
            f"vertical_benchmarks.yaml is empty or malformed at {benchmarks_file}."
        )
    return data