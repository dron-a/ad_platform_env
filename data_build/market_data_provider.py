"""
ETL class that sources real-world advertising market data and converts
it into CampaignProfile objects for the AdPlatform RL environment.

Data hierarchy (lowest to highest priority):
    Synthetic defaults
    → MarketDataProvider (Google Trends / Wikipedia / FRED)
    → user YAML (yaml_path at env startup via ProfileLoader)
    → runtime CampaignProfile (passed to reset() per episode)

Data sources (priority chain, falls back on failure):
    1. Google Trends (live)   — via pytrends, reflects real search interest
    2. Google Trends (cached) — on-disk cache, max 7 days old
    3. Wikipedia (live)       — pageview API, no auth, highly reliable
    4. Wikipedia (cached)     — on-disk cache, max 7 days old
    5. FRED (live)            — macro economic indicators, vertical-specific
    6. FRED (cached)          — on-disk cache, max 7 days old
    7. default_profile.yaml   — always works, offline fallback
    8. vertical benchmarks    — hardcoded WordStream benchmarks, last resort


Vertical keyword mapping:
    Each vertical maps three keywords to three campaign types:
        Campaign 0: Branded search  — high intent, best CVR
        Campaign 1: Generic search  — medium intent, moderate CVR
        Campaign 2: Display/retarg  — broad reach, lowest CVR
"""

from __future__ import annotations

import datetime
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import keyword mappings and tuning constants from market_constants.py
# Import benchmark data from vertical_benchmarks.yaml
# ---------------------------------------------------------------------------

from .market_constants import (
    VERTICAL_KEYWORDS,
    WIKIPEDIA_TITLES,
    EPISODE_WINDOW_DAYS,
    DATASET_LOOKBACK_DAYS,
    SLIDE_STEP_DAYS,
    SPIKE_THRESHOLD,
    CACHE_EXPIRY_DAYS,
    WIKIPEDIA_MAX_RETRIES,
    _load_vertical_benchmarks
)



# Load once at module level — available to all methods
VERTICAL_BENCHMARKS: Dict = _load_vertical_benchmarks()


# ---------------------------------------------------------------------------
# MarketDataProvider
# ---------------------------------------------------------------------------

class MarketDataProvider:
    """
    Sources real-world advertising market data and converts it into
    CampaignProfile objects for the AdPlatform RL environment.

    Parameters
    ----------
    vertical          : str — "ecommerce" | "saas" | "travel" | "finance"
    refresh_per_episode: bool — if True, sample new window each episode
    sampling          : str — "sequential" | "random" (only when refresh=True)
    seed              : int — random seed for reproducibility
    cache_dir         : str — directory to cache fetched data
    """


    # Class-level shared cache — one dataset per vertical per process
    # Ensures multiple sessions for same vertical share one network fetch
    _shared_datasets: dict = {}
    _shared_sources:  dict = {}
    _shared_dates:    dict = {}

    def __init__(
        self,
        vertical:             str  = "ecommerce",
        refresh_per_episode:  bool = True,
        sampling:             str  = "sequential",
        seed:                 int  = 42,
        cache_dir:            str  = ".market_cache",
    ):
        if vertical not in VERTICAL_KEYWORDS:
            raise ValueError(
                f"Unknown vertical: '{vertical}'. "
                f"Valid: {list(VERTICAL_KEYWORDS.keys())}"
            )
        if sampling not in ("sequential", "random"):
            raise ValueError(
                f"Unknown sampling: '{sampling}'. Valid: 'sequential', 'random'"
            )

        self.vertical            = vertical
        self.refresh_per_episode = refresh_per_episode
        self.sampling            = sampling
        self.seed                = seed
        self.cache_dir           = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self._rng                 = random.Random(seed)
        self._np_rng              = np.random.default_rng(seed)
        self._episode_count       = 0
        self._dataset: Optional[Dict] = None  # fetched at startup
        self._source_used         = "none"
        self._data_date_range     = "unknown"

        # Use shared cache if already fetched for this vertical — no network call
        if vertical in MarketDataProvider._shared_datasets:
            self._dataset         = MarketDataProvider._shared_datasets[vertical]
            self._source_used     = MarketDataProvider._shared_sources[vertical]
            self._data_date_range = MarketDataProvider._shared_dates[vertical]
            logger.info(f"MarketDataProvider: reusing shared cache for {vertical}")
        else:
            self._dataset = self._fetch_with_fallback()
            if self._dataset is not None:
                MarketDataProvider._shared_datasets[vertical] = self._dataset
                MarketDataProvider._shared_sources[vertical]  = self._source_used
                MarketDataProvider._shared_dates[vertical]    = self._data_date_range

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Which data source was successfully used."""
        return self._source_used

    @property
    def data_date_range(self) -> str:
        """Date range of the fetched dataset."""
        return self._data_date_range

    def get_profile(self) -> "CampaignProfile":
        """
        Return a CampaignProfile for the next episode.

        Mode 1 (refresh=False): always returns the same profile
        Mode 2 (sequential):    slides window forward each call
        Mode 3 (random):        random window each call
        """
        # Import here to avoid circular imports
        try:
            from ..models import CampaignProfile
        except ImportError:
            from models import CampaignProfile

        if self._dataset is None:
            logger.warning("MarketDataProvider: no dataset, returning fallback profile")
            return self._fallback_profile(CampaignProfile)

        window = self._select_window()
        profile = self._build_profile(window, CampaignProfile)

        if self.refresh_per_episode:
            self._episode_count += 1

        return profile

    def reset_episode_counter(self) -> None:
        """Reset sequential window position to start."""
        self._episode_count = 0

    def set_vertical(self, vertical: str) -> None:
        """
        Switch to a different vertical without recreating the provider.
        Re-fetches data for the new vertical and resets episode counter.

        Parameters
        ----------
        vertical : str — "ecommerce" | "saas" | "travel" | "finance"
        """
        if vertical not in VERTICAL_KEYWORDS:
            raise ValueError(
                f"Unknown vertical: '{vertical}'. "
                f"Valid: {list(VERTICAL_KEYWORDS.keys())}"
            )
        if vertical == self.vertical:
            return   # no change needed

        self.vertical       = vertical
        self._episode_count = 0

        if vertical in MarketDataProvider._shared_datasets:
            self._dataset         = MarketDataProvider._shared_datasets[vertical]
            self._source_used     = MarketDataProvider._shared_sources[vertical]
            self._data_date_range = MarketDataProvider._shared_dates[vertical]
            logger.info(f"MarketDataProvider: switched to {vertical} (from shared cache)")
        else:
            self._dataset = self._fetch_with_fallback()
            if self._dataset is not None:
                MarketDataProvider._shared_datasets[vertical] = self._dataset
                MarketDataProvider._shared_sources[vertical]  = self._source_used
                MarketDataProvider._shared_dates[vertical]    = self._data_date_range
            logger.info(f"MarketDataProvider: switched to {vertical} (fetched fresh, source={self._source_used})")

    def refresh(self) -> bool:
        """
        Re-fetch data from sources for the current vertical.
        Call periodically for long-running training to get fresh market data.
        Resets sequential window position on successful refresh.

        Returns True if data was successfully refreshed, False otherwise.
        """
        new_data = self._fetch_with_fallback()
        if new_data is not None:
            self._dataset       = new_data
            self._episode_count = 0
            # Update shared cache so future instances benefit from fresh data
            MarketDataProvider._shared_datasets[self.vertical] = new_data
            MarketDataProvider._shared_sources[self.vertical]  = self._source_used
            MarketDataProvider._shared_dates[self.vertical]    = self._data_date_range
            logger.info(
                f"MarketDataProvider: refreshed {self.vertical} from "
                f"{self._source_used} ({self._data_date_range})"
            )
            return True
        logger.warning(
            "MarketDataProvider: refresh failed, keeping existing data"
        )
        return False

    # -----------------------------------------------------------------------
    # Data fetching — priority chain
    # Live source tried first, then its cache, then next live source, etc.
    # -----------------------------------------------------------------------

    def _fetch_with_fallback(self) -> Optional[Dict]:
        """
        Try each data source in priority order.

        Order:
          1. Google Trends (live)
          2. Google Trends (cached)
          3. Wikipedia (live)
          4. Wikipedia (cached)
          5. FRED (live)
          6. FRED (cached)
          7. Returns None → get_profile() uses fallback_profile()

        Returns normalized dataset dict or None if all sources fail.
        """
        sources = [
            ("Google Trends", self._fetch_google_trends),
            ("Wikipedia",     self._fetch_wikipedia),
            ("FRED",          self._fetch_fred),
        ]

        for name, fetch_fn in sources:
            # Try live source first
            try:
                logger.info(f"MarketDataProvider: trying {name} (live)...")
                data = fetch_fn()
                if data is not None and self._validate_dataset(data):
                    self._source_used = name
                    self._update_date_range(data)
                    logger.info(
                        f"MarketDataProvider: using {name} "
                        f"({self._data_date_range})"
                    )
                    print(f"[MarketDataProvider] using {name} ({self._data_date_range})", flush=True)
                    self._cache_dataset(data, name)
                    return data
            except Exception as e:
                logger.warning(f"MarketDataProvider: {name} (live) failed — {e}")
                print(f"[MarketDataProvider] {name} failed — {e}", flush=True)

            # Try cached source before falling to next live source
            cached = self._load_cached(name)
            if cached is not None and self._validate_dataset(cached):
                self._source_used = f"{name} (cached)"
                self._update_date_range(cached)
                logger.info(
                    f"MarketDataProvider: using {name} (cached) "
                    f"({self._data_date_range})"
                )
                return cached
            elif cached is not None:
                logger.debug(f"MarketDataProvider: {name} cache invalid or stale")

        self._source_used = "default_profile.yaml"
        logger.warning(
            "MarketDataProvider: all sources failed — "
            "using default_profile.yaml fallback"
        )
        return None

    def _fetch_google_trends(self) -> Optional[Dict]:
        """
        Fetch interest over time from Google Trends via pytrends.
        Returns normalized series dict or None.
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            raise ImportError("pytrends not installed: pip install pytrends")

        keywords = VERTICAL_KEYWORDS[self.vertical]
        kw_list  = [
            keywords["branded"],
            keywords["generic"],
            keywords["display"],
        ]

        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(
            kw_list,
            timeframe=f"today {DATASET_LOOKBACK_DAYS}-d",
            geo="US",
        )
        # Rate limit courtesy
        time.sleep(1.0)

        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return None

        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        # Normalise each keyword series to multipliers around 1.0
        result = {}
        for i, kw in enumerate(kw_list):
            if kw not in df.columns:
                continue
            series = df[kw].values.astype(float)
            mean   = series.mean() + 1e-8
            result[f"campaign_{i}"] = (series / mean).tolist()

        result["dates"] = [str(d.date()) for d in df.index]
        return result if len(result) >= 4 else None   # 3 campaigns + dates

    def _fetch_wikipedia(self) -> Optional[Dict]:
        """
        Fetch Wikipedia pageview data for vertical keywords.
        Uses Wikimedia REST API — no auth required.
        Includes retry logic with exponential backoff.
        """
        try:
            import urllib.request
            import json as _json
        except ImportError:
            return None

        titles   = WIKIPEDIA_TITLES.get(self.vertical, [])
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=DATASET_LOOKBACK_DAYS)

        result = {}
        for i, title in enumerate(titles[:3]):
            url = (
                f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
                f"per-article/en.wikipedia/all-access/all-agents/"
                f"{title}/daily/"
                f"{start_date.strftime('%Y%m%d')}/"
                f"{end_date.strftime('%Y%m%d')}"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "AdPlatformRL/1.0"}
            )

            # Retry with exponential backoff
            last_exc = None
            for attempt in range(WIKIPEDIA_MAX_RETRIES):
                try:
                    resp  = urllib.request.urlopen(req, timeout=10)
                    data  = _json.loads(resp.read())
                    views = [item["views"] for item in data.get("items", [])]
                    if views:
                        arr  = np.array(views, dtype=float)
                        mean = arr.mean() + 1e-8
                        result[f"campaign_{i}"] = (arr / mean).tolist()
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < WIKIPEDIA_MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        logger.debug(
                            f"Wikipedia retry {attempt+1}/{WIKIPEDIA_MAX_RETRIES} "
                            f"for {title} — waiting {wait}s"
                        )
                        time.sleep(wait)

            if f"campaign_{i}" not in result and last_exc:
                logger.debug(f"Wikipedia failed for {title}: {last_exc}")

            time.sleep(0.5)   # courtesy delay between articles

        if len(result) < 3:
            return None

        # Align lengths — trim to shortest
        min_len = min(len(v) for v in result.values())
        for k in result:
            result[k] = result[k][:min_len]

        result["dates"] = [
            str(start_date + datetime.timedelta(days=i))
            for i in range(min_len)
        ]
        return result

    def _fetch_fred(self) -> Optional[Dict]:
        """
        Fetch FRED macro indicator as a market-wide signal.
        Handles FRED's missing value marker ('.') correctly.
        Applies vertical-specific campaign dampening.
        """
        import urllib.request
        import json as _json

        fred_series = VERTICAL_KEYWORDS[self.vertical]["fred"]
        api_key     = os.getenv("FRED_API_KEY", "")   # optional — public endpoint also works

        end_date   = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=DATASET_LOOKBACK_DAYS * 2)

        try:
            if api_key:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={fred_series}"
                    f"&observation_start={start_date}"
                    f"&observation_end={end_date}"
                    f"&api_key={api_key}&file_type=json"
                )
            else:
                url = (
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                    f"?id={fred_series}"
                    f"&vintage_date={end_date}"
                )

            req  = urllib.request.Request(
                url, headers={"User-Agent": "AdPlatformRL/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=15)
            raw  = resp.read().decode("utf-8")

            values = []
            dates  = []

            if api_key:
                data = _json.loads(raw)
                for obs in data.get("observations", []):
                    # FRED uses "." for missing values — skip them
                    val_str = obs.get("value", ".").strip()
                    if val_str == ".":
                        continue
                    try:
                        values.append(float(val_str))
                        dates.append(obs["date"])
                    except (ValueError, KeyError):
                        continue
            else:
                for line in raw.strip().split("\n")[1:]:
                    parts = line.split(",")
                    if len(parts) != 2:
                        continue
                    val_str = parts[1].strip()
                    # Skip FRED missing value marker and empty strings
                    if val_str in (".", "", "NA", "N/A"):
                        continue
                    try:
                        values.append(float(val_str))
                        dates.append(parts[0].strip())
                    except ValueError:
                        continue

            if len(values) < EPISODE_WINDOW_DAYS:
                return None

            arr  = np.array(values[-DATASET_LOOKBACK_DAYS:], dtype=float)
            mean = arr.mean() + 1e-8
            norm = (arr / mean).tolist()
            dates = dates[-DATASET_LOOKBACK_DAYS:]

            # Campaign-specific dampening:
            # Campaign 0 (branded) — full macro signal
            # Campaign 1 (generic) — moderately dampened
            # Campaign 2 (display) — most dampened (least sensitive to macro)
            result = {
                "campaign_0": [float(v * 1.00) for v in norm],
                "campaign_1": [float(v * 0.85) for v in norm],
                "campaign_2": [float(v * 0.60) for v in norm],
                "dates":      dates,
            }
            return result

        except Exception as e:
            logger.debug(f"FRED fetch failed: {e}")
            return None

    # -----------------------------------------------------------------------
    # Window selection
    # -----------------------------------------------------------------------

    def _select_window(self) -> Dict:
        """
        Select a 30-day window from the dataset based on current mode.

        Mode 1 (refresh=False): always window starting at 0
        Mode 2 (sequential):    slide by SLIDE_STEP_DAYS per episode, wrap
        Mode 3 (random):        random start position each call

        Edge case: if dataset is exactly EPISODE_WINDOW_DAYS long,
        max_start=0 and start is always 0 regardless of mode — effectively
        Mode 1 behaviour. Documented here intentionally.
        """
        n_points = len(self._dataset.get("campaign_0", []))
        max_start = max(0, n_points - EPISODE_WINDOW_DAYS)

        if not self.refresh_per_episode:
            # Mode 1 — fixed
            start = 0
        elif self.sampling == "sequential":
            # Mode 2 — slide forward SLIDE_STEP_DAYS per episode, wrap
            start = (self._episode_count * SLIDE_STEP_DAYS) % max(max_start, 1)
        else:
            # Mode 3 — random
            start = self._rng.randint(0, max_start)

        end = start + EPISODE_WINDOW_DAYS

        window = {}

        for k, v in self._dataset.items():
            if k == "dates":
                window["dates"] = v[start:end] if start < len(v) else v[:EPISODE_WINDOW_DAYS]
            else:
                series = list(v[start:end])
                # Pad with last value if window extends beyond data
                while len(series) < EPISODE_WINDOW_DAYS:
                    series.append(series[-1] if series else 1.0)
                window[k] = series

        return window

    # -----------------------------------------------------------------------
    # Profile building
    # -----------------------------------------------------------------------

    def _build_profile(self, window: Dict, CampaignProfile) -> "CampaignProfile":
        """
        Convert a 30-day data window into a CampaignProfile.

        Steps:
          1. Seasonal multipliers — averaged across all three campaigns
             (more representative than branded signal alone)
          2. Competition index — mean trend level per campaign this window
          3. Conversion rates — base CVR scaled by sqrt(competition index)
          4. Competitor bids — base CPC scaled linearly by competition index
          5. Bid volatility — computed from std dev / mean of trend window
          6. Market events — detected from z-score spikes, top 3 by significance
        """
        benchmarks = VERTICAL_BENCHMARKS[self.vertical]

        # --- Seasonal multipliers — direct from window ---
        # Use campaign_0 (branded/highest quality signal) as primary
        seasonal = window.get("campaign_0", [1.0] * EPISODE_WINDOW_DAYS)
        # Clamp to reasonable range [0.3, 3.0]
        seasonal = [float(max(0.3, min(3.0, v))) for v in seasonal]

        # --- Competition index — mean trend level for this window ---
        # High search interest → more competitors → higher effective CPCs
        c0 = window.get("campaign_0", [1.0] * EPISODE_WINDOW_DAYS)
        c1 = window.get("campaign_1", [1.0] * EPISODE_WINDOW_DAYS)
        c2 = window.get("campaign_2", [1.0] * EPISODE_WINDOW_DAYS)

        # --- Seasonal multipliers ---
        # Average across all three campaigns — more representative of overall
        # market activity than using branded signal alone. Clamp to [0.3, 3.0].
        n = len(c0)
        seasonal = [
            float(max(0.3, min(3.0, (c0[i] + c1[i] + c2[i]) / 3.0)))
            for i in range(n)
        ]

        # --- Competition index per campaign ---
        comp_index_0 = float(np.mean(c0))
        comp_index_1 = float(np.mean(c1))
        comp_index_2 = float(np.mean(c2))

        # --- Conversion rates --- 
        # Scale base CVR by competition index
        # High competition → more qualified traffic → slightly higher CVR
        # but also more expensive → use sqrt to dampen effect
        base_cvrs = benchmarks["conversion_rates"]
        conversion_rates = [
            float(round(base_cvrs[0] * np.sqrt(max(comp_index_0, 0.5)), 5)),
            float(round(base_cvrs[1] * np.sqrt(max(comp_index_1, 0.5)), 5)),
            float(round(base_cvrs[2] * np.sqrt(max(comp_index_2, 0.5)), 5)),
        ]

        # --- Competitor bids ---
        # Scale base CPC linearly by competition index
        # High interest → more competitors → higher auction clearing price
        # Linear scaling — CPC responds directly to competition level
        base_cpcs = benchmarks["competitor_bids"]
        competitor_bids = [
            float(round(base_cpcs[0] * max(comp_index_0, 0.5), 4)),
            float(round(base_cpcs[1] * max(comp_index_1, 0.5), 4)),
            float(round(base_cpcs[2] * max(comp_index_2, 0.5), 4)),
        ]

        # --- Bid volatility ---
        # Coefficient of variation of trend window — volatile markets = volatile bids
        vol_0 = float(np.std(c0)) / (float(np.mean(c0)) + 1e-8)
        vol_1 = float(np.std(c1)) / (float(np.mean(c1)) + 1e-8)
        vol_2 = float(np.std(c2)) / (float(np.mean(c2)) + 1e-8)
        bid_volatility = [
            float(round(max(0.03, min(0.30, vol_0)), 4)),
            float(round(max(0.03, min(0.30, vol_1)), 4)),
            float(round(max(0.03, min(0.30, vol_2)), 4)),
        ]

        # --- Market events ---
        # Detect spikes above SPIKE_THRESHOLD std devs in branded signal
        # These represent real market events — product launches, news cycles, etc.
        market_events = self._detect_market_events(c0, c1, c2)

        # Log profile summary for transparency
        dates = window.get("dates", [])
        date_range = f"{dates[0]} to {dates[-1]}" if dates else "unknown"
        logger.info(
            f"MarketDataProvider: built profile for {date_range} | "
            f"cvr={conversion_rates} | cpcs={competitor_bids} | "
            f"events={list(market_events.keys())}"
        )

        return CampaignProfile(
            conversion_rates     = conversion_rates,
            competitor_bids      = competitor_bids,
            bid_volatility       = bid_volatility,
            seasonal_multipliers = seasonal,
            market_events        = market_events,
            total_budget         = float(benchmarks["total_budget"]),
        )

    def _detect_market_events(
        self,
        c0: List[float],
        c1: List[float],
        c2: List[float],
    ) -> Dict[int, List[float]]:
        """
        Detect market event steps from trend spikes.

        A step is a market event if any campaign's z-score exceeds
        SPIKE_THRESHOLD. Keeps top 3 events by z-score significance
        (not by multiplier magnitude) to select truly anomalous periods.

        Returns dict mapping step index to per-campaign multipliers.
        Max 3 events per episode to avoid noise overwhelming the signal.
        """
        arr0 = np.array(c0)
        arr1 = np.array(c1)
        arr2 = np.array(c2)

        mean0, std0 = arr0.mean(), arr0.std() + 1e-8
        mean1, std1 = arr1.mean(), arr1.std() + 1e-8
        mean2, std2 = arr2.mean(), arr2.std() + 1e-8

        # Store (multipliers, max_zscore) per event step
        candidates = {}
        for step in range(len(c0)):
            z0 = (arr0[step] - mean0) / std0
            z1 = (arr1[step] - mean1) / std1
            z2 = (arr2[step] - mean2) / std2
            max_z = max(z0, z1, z2)

            if max_z >= SPIKE_THRESHOLD:
                mult0 = float(max(0.5, min(2.5, arr0[step] / (mean0 + 1e-8))))
                mult1 = float(max(0.5, min(2.5, arr1[step] / (mean1 + 1e-8))))
                mult2 = float(max(0.5, min(2.5, arr2[step] / (mean2 + 1e-8))))
                candidates[step] = {
                    "multipliers": [
                        round(mult0, 2),
                        round(mult1, 2),
                        round(mult2, 2),
                    ],
                    "zscore": float(max_z),
                }

        # Keep top 3 by z-score significance — most anomalous periods
        if len(candidates) > 3:
            top3 = sorted(
                candidates.items(),
                key=lambda x: x[1]["zscore"],
                reverse=True,
            )[:3]
            candidates = dict(top3)

        # Return only multipliers — zscore was for selection only
        return {
            step: info["multipliers"]
            for step, info in candidates.items()
        }

    # -----------------------------------------------------------------------
    # Fallback profile
    # -----------------------------------------------------------------------

    def _fallback_profile(self, CampaignProfile) -> "CampaignProfile":
        """
        Return vertical benchmark profile when all data sources fail.

        Uses VERTICAL_BENCHMARKS which includes full profile fields per
        vertical — conversion_rates, competitor_bids, bid_volatility,
        seasonal_multipliers, total_budget.

        default_profile.yaml belongs to ProfileLoader (user's own data path)
        not here — MarketDataProvider always works within a vertical context.
        """
        # Vertical benchmarks — always
        benchmarks = VERTICAL_BENCHMARKS[self.vertical]
        logger.info(
            f"MarketDataProvider: fallback using {self.vertical} benchmarks"
        )
        return CampaignProfile(**benchmarks)

    # -----------------------------------------------------------------------
    # Caching — with staleness check
    # -----------------------------------------------------------------------

    def _cache_dataset(self, data: Dict, source: str) -> None:
        """Cache fetched dataset to disk for offline fallback."""
        try:
            cache_file = self._cache_path(source)
            with open(cache_file, "w") as f:
                yaml.dump(data, f)
            logger.debug(f"MarketDataProvider: cached {source} to {cache_file}")
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _load_cached(self, source: str) -> Optional[Dict]:
        """
        Load cached dataset if available and not stale.
        Cache older than CACHE_EXPIRY_DAYS is treated as stale and ignored.
        """
        try:
            cache_file = self._cache_path(source)
            if not cache_file.exists():
                return None

            # Check cache age
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days > CACHE_EXPIRY_DAYS:
                logger.debug(
                    f"Cache stale ({age_days:.1f} days > {CACHE_EXPIRY_DAYS}), "
                    f"ignoring {source} cache"
                )
                return None

            with open(cache_file) as f:
                data = yaml.safe_load(f)
            logger.debug(
                f"MarketDataProvider: loaded {source} cache "
                f"(age {age_days:.1f} days)"
            )
            return data

        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        return None

    def _cache_path(self, source: str) -> Path:
        """Return cache file path for a given source."""
        return self.cache_dir / f"{self.vertical}_{source.lower().replace(' ', '_')}.yaml"

    # -----------------------------------------------------------------------
    # Validation and utilities
    # -----------------------------------------------------------------------

    def _validate_dataset(self, data: Dict) -> bool:
        """Validate that dataset has minimum required structure."""
        if not isinstance(data, dict):
            return False
        for key in ("campaign_0", "campaign_1", "campaign_2"):
            if key not in data:
                return False
            if len(data[key]) < EPISODE_WINDOW_DAYS:
                return False
        return True

    def _update_date_range(self, data: Dict) -> None:
        """Update internal date range string from dataset dates."""
        dates = data.get("dates", [])
        if dates:
            self._data_date_range = f"{dates[0]} to {dates[-1]}"
        else:
            self._data_date_range = "unknown"

    # -----------------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MarketDataProvider("
            f"vertical={self.vertical!r}, "
            f"source={self._source_used!r}, "
            f"dates={self._data_date_range!r}, "
            f"refresh_per_episode={self.refresh_per_episode}, "
            f"sampling={self.sampling!r}, "
            f"episode={self._episode_count}"
            f")"
        )