"""
profile_loader.py
=================
Handles loading real world Ads export data from YAML and merging with
runtime CampaignProfile overrides for this project, can be set during training too

Three-tier data hierarchy (lowest to highest priority):
  Tier 1 — Synthetic defaults  : AdPlatformState field defaults
  Tier 2 — YAML file           : loaded once at startup via ProfileLoader
  Tier 3 — Runtime profile     : CampaignProfile passed to reset() per episode

All tiers are additive — only fields explicitly provided override lower tiers.
Anything not provided rolls down to the next tier.

Usage
-----
# Startup — load once
loader = ProfileLoader("ads_export.yaml")

# Training loop — YAML only
env.reset(task="auction", profile=loader.profile)

# Training loop — YAML base + episode override
episode_profile = CampaignProfile(total_budget=3000.0)
env.reset(task="auction", profile=ProfileLoader.resolve(loader.profile, episode_profile))

# Training loop — no file, runtime only
env.reset(task="auction", profile=CampaignProfile(total_budget=2000.0))

# Training loop — pure synthetic, no profile
env.reset(task="auction")

YAML format
-----------
# google_ads_export.yaml
conversion_rates: [0.04, 0.02, 0.015]
competitor_bids: [0.80, 0.60, 0.45]
bid_volatility: [0.08, 0.12, 0.06]
seasonal_multipliers: [1.3, 1.1, 0.9, 1.2, 1.4, 1.5, 0.8]
market_events:
  3: [1.5, 1.0, 0.9]
  15: [1.0, 1.3, 1.0]
total_budget: 5000.0

All fields are optional. Any omitted field falls back to synthetic defaults.
Field names match CampaignProfile exactly — same validation applies.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from models import CampaignProfile


class ProfileLoader:
    """
    Loads a YAML campaign data file once at startup and exposes it as a
    validated CampaignProfile. Provides a resolve() method to merge the
    loaded profile with a per-episode runtime override.

    Parameters
    ----------
    yaml_path : str | Path | None
        Path to the YAML file. If None, no file is loaded and
        loader.profile is None — pure synthetic defaults will be used.
    """

    def __init__(self, yaml_path: str | Path | None = None):
        self._yaml_path = Path(yaml_path) if yaml_path is not None else None
        self._profile: CampaignProfile | None = None

        if self._yaml_path is not None:
            self._profile = self._load(self._yaml_path)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    @property
    def profile(self) -> CampaignProfile | None:
        """
        The loaded and validated CampaignProfile from the YAML file.
        None if no file was provided.
        """
        return self._profile

    @staticmethod
    def resolve(
        yaml_profile:    CampaignProfile | None,
        runtime_profile: CampaignProfile | None,
    ) -> CampaignProfile | None:
        """
        Merge yaml_profile and runtime_profile into a single CampaignProfile.

        Priority: runtime_profile fields WIN over yaml_profile fields.
        Only fields explicitly present in either profile are included —
        anything absent rolls down to synthetic defaults in apply_profile.

        Parameters
        ----------
        yaml_profile    : loaded from YAML file via ProfileLoader
        runtime_profile : passed to reset() per episode

        Returns
        -------
        Merged CampaignProfile, or None if both inputs are None/empty.
        """
        if not yaml_profile and not runtime_profile:
            return None

        merged = {}

        # Start with YAML base
        if yaml_profile:
            merged.update(yaml_profile)

        # Runtime wins field by field
        if runtime_profile:
            merged.update(runtime_profile)

        return CampaignProfile(**merged) if merged else None

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> CampaignProfile:
        """
        Load, parse, and validate a YAML file into a CampaignProfile.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Campaign profile YAML not found: {path}. "
                f"Check the file path or omit it to use synthetic defaults."
            )

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(
                f"Campaign profile YAML is empty: {path}. "
                f"Provide at least one field or omit the file."
            )

        if not isinstance(raw, dict):
            raise ValueError(
                f"Campaign profile YAML must be a mapping of field: value pairs. "
                f"Got: {type(raw)} in {path}"
            )

        # market_events keys come from YAML as integers already if written
        # as bare integers, but yaml.safe_load may parse them as int or str
        # depending on formatting. Normalise to int here before validation.
        if "market_events" in raw and isinstance(raw["market_events"], dict):
            raw["market_events"] = {
                int(k): v for k, v in raw["market_events"].items()
            }

        # Pass through CampaignProfile for full validation
        try:
            return CampaignProfile(**raw)
        except (TypeError, ValueError) as e:
            raise type(e)(
                f"Invalid campaign profile in {path}: {e}"
            ) from e
