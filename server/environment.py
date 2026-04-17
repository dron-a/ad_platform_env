import numpy as np
from typing import Literal
from openenv.core.env_server import Environment, State
from pathlib import Path
import server.tasks as tasks
from .profile_loader import ProfileLoader

try:
    from ..models import AdPlatformState, AdPlatformAction, CampaignProfile
except ImportError:
    from models import AdPlatformState, AdPlatformAction, CampaignProfile

DEFAULT_YAML = Path(__file__).parent / "default_profile.yaml"


class AdPlatformEnvironment(Environment):
    """
    Unified environment supporting three tasks:
      1. budget           — allocate fixed budget to maximize conversions
      2. auction          — bid competitively against adaptive competitors
      3. dynamic_campaign — manage budget and bids with seasonality and market events

    Profile data hierarchy (lowest to highest priority):
      Tier 1 — Synthetic defaults  : AdPlatformState field defaults
      Tier 2 — default_profile.yaml: loaded via ProfileLoader when no yaml_path set
      Tier 3 — YAML file           : user's own data via yaml_path
      Tier 4 — MarketDataProvider  : real market data per episode via vertical
      Tier 5 — Runtime profile     : CampaignProfile passed to reset() per episode

    Parameters
    ----------
    task      : starting task — can be switched at reset time
    yaml_path : path to Ads export YAML file (optional)
    vertical  : market vertical for real-world data — "ecommerce" | "saas" |
                "travel" | "finance" (optional). Creates MarketDataProvider.
                Can be overridden per episode via reset(vertical=...).
    """

    state_type = AdPlatformState

    def __init__(
        self,
        task:      Literal["budget", "auction", "dynamic_campaign"] = "budget",
        yaml_path: str | None = None,
        vertical:  str | None = None,
        _market_provider: object | None = None,   # internal — pre-fetched
    ):
        super().__init__()
        self.task   = task
        self._state = AdPlatformState()

        # Load YAML profile once at startup
        # Load default_profile.yaml as base — always present if file exists
        # Use provided yaml_path, fall back to default_profile.yaml as safeguard
        DEFAULT_YAML = Path(__file__).parent / "default_profile.yaml"
        self._default_loader = ProfileLoader(str(DEFAULT_YAML) if DEFAULT_YAML.exists() else None)

        # Load user YAML separately — None if not provided
        self._loader = ProfileLoader(yaml_path)

        # MarketDataProvider — _market_provider is internal,
        # If pre-fetched provider passed use it directly — no network call
        # Otherwise create from vertical if provided
        if _market_provider is not None:
            self._market_provider = _market_provider
        elif vertical is not None:
            try:
                from ..data_build import MarketDataProvider
            except ImportError:
                from data_build import MarketDataProvider
            self._market_provider = MarketDataProvider(vertical=vertical)

    # ---------------- RESET ----------------

    def reset(
        self,
        task:         Literal["budget", "auction", "dynamic_campaign"] | None = None,
        realism_mode: str | None = None,
        profile:      CampaignProfile | None = None,
        vertical:     str | None = None,
    ):
        """
        Reset the environment for a new episode.

        Args:
            task         : switch active task (optional)
            realism_mode : "fixed" or "realistic" conversion rate mode
                           (budget task only)
            profile      : CampaignProfile with real data to inject this episode
                           wins over all other sources field by field
            vertical     : switch market vertical for this and future episodes
                           creates MarketDataProvider if not already present
                           if None uses last set vertical or no market data
        """
        s = self._state

        if task is not None:
            self.task = task

        # Handle vertical switching — creates or switches MarketDataProvider
        if vertical is not None:
            if self._market_provider is None:
                try:
                    from ..data_build import MarketDataProvider
                except ImportError:
                    from data_build import MarketDataProvider
                self._market_provider = MarketDataProvider(vertical=vertical)
            elif self._market_provider.vertical != vertical:
                self._market_provider.set_vertical(vertical)

        # Resolve four-tier profile hierarchy — invisible to agent
        # default_profile.yaml → market provider → user YAML → runtime profile
        market_profile = (
            self._market_provider.get_profile()
            if self._market_provider is not None
            else None
        )
        after_default = ProfileLoader.resolve(self._default_loader.profile, market_profile)
        after_market  = ProfileLoader.resolve(after_default, self._loader.profile)
        resolved      = ProfileLoader.resolve(after_market, profile)

        if self.task == "budget":
            return tasks.reset_budget(s, realism_mode=realism_mode, profile=resolved)
        elif self.task == "auction":
            return tasks.reset_auction(s, profile=resolved)
        elif self.task == "dynamic_campaign":
            return tasks.reset_dynamic_campaign(s, profile=resolved)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    # ---------------- STEP ----------------

    def step(self, action: AdPlatformAction):
        """Step the environment depending on the active task."""
        s = self._state

        # Guard against stepping a terminated episode
        if s.step_count >= s.max_steps or s.remaining_budget <= 0.0:
            raise ValueError(
                f"Episode already terminated at step {s.step_count} "
                f"(max_steps={s.max_steps}, remaining_budget={s.remaining_budget:.2f}). "
                f"Call reset() to start a new episode."
            )

        if self.task == "budget":
            return tasks.step_budget(self._state, action)
        elif self.task == "auction":
            return tasks.step_auction(self._state, action)
        elif self.task == "dynamic_campaign":
            return tasks.step_dynamic_campaign(self._state, action)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    # ---------------- STATE ----------------

    @property
    def state(self):
        s = self._state
        if self.task == "budget":
            return State(
                episode_id           = None,
                step_count           = s.step_count,
                remaining_budget     = s.remaining_budget,
                campaign_performance = s.conversion_rates.copy(),
                reward_buffer        = list(s.reward_buffer),  # has to be JSON-serializable
                total_conversions    = s.total_conversions,
                total_spend          = s.total_spend,
                total_budget         = s.total_budget,
                done                 = s.step_count >= s.max_steps or s.remaining_budget <= 0.0,
                vertical             = self._market_provider.vertical
                                     if self._market_provider else None,
            )
        else:
            return State(
                episode_id        = None,
                step_count        = s.step_count,
                remaining_budget  = s.remaining_budget,
                competitor_bids   = getattr(s, "competitor_bids",  []),
                conversion_rates  = getattr(s, "conversion_rates", []),
                total_conversions = getattr(s, "total_conversions", 0.0),
                total_spend       = getattr(s, "total_spend",       0.0),
                total_budget      = s.total_budget,
                done              = s.step_count >= s.max_steps or s.remaining_budget <= 0.0,
                vertical          = self._market_provider.vertical
                                    if self._market_provider else None,
            )

    # ---------------- MAX SCORE ----------------

    @property
    def max_possible_conversions(self) -> float:
        s = self._state
        return s.total_budget * max(getattr(s, "conversion_rates", [1.0]))