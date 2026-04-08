# server/environment.py

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
      Tier 2 — YAML file           : loaded once at startup via yaml_path
      Tier 3 — Runtime profile     : CampaignProfile passed to reset() per episode

    Parameters
    ----------
    task      : starting task — can be switched at reset time
    yaml_path : path to Ads export YAML file (optional)
                if None, synthetic defaults are used unless runtime profile provided
    """

    state_type = AdPlatformState

    def __init__(
        self,
        task: Literal["budget", "auction", "dynamic_campaign"] = "budget",
        yaml_path: str | None = None,
    ):
        super().__init__()
        self.task = task
        self._state = AdPlatformState()

        # Load YAML profile once at startup — None if no file provided
        # Use provided yaml_path, fall back to default_profile.yaml as safeguard
        # If neither exists, ProfileLoader returns None and synthetic defaults are used
        resolved_path = yaml_path or (str(DEFAULT_YAML) if DEFAULT_YAML.exists() else None)
        self._loader  = ProfileLoader(resolved_path)
        # self._loader = ProfileLoader(yaml_path)


        # self.state_type = AdPlatformState

        # self._state = None
        # if task == "budget":
        #     self._state = BudgetState()
        #     self.state_type = BudgetState
        # elif task == "auction":
        #     self._state = AuctionState()
        #     self.state_type = AuctionState
        # elif task == "dynamic_campaign":
        #     self._state = DynamicState()
        #     self.state_type = DynamicState
        # else:
        #     raise ValueError(f"Unknown task: {task}")

    # ---------------- RESET ----------------
    def reset(self, task: Literal["budget", "auction", "dynamic_campaign"] = None,
              realism_mode: str | None = None, profile: CampaignProfile | None = None):
        """
        Reset the environment depending on the task.

        Args:
            task:         Switch active task (optional).
            realism_mode: "fixed" or "realistic" conversion rate mode (budget task only).
            profile:      CampaignProfile dict with real historical data to inject.
                          Overrides built-in defaults for conversion_rates,
                          competitor_bids, bid_volatility, seasonal_multipliers,
                          market_events, and total_budget.
        """
        s = self._state
        if task is not None:
            self.task = task

        # Resolve three-tier profile hierarchy here — invisible to agent
        # Runtime profile wins over YAML, YAML wins over synthetic defaults
        resolved = ProfileLoader.resolve(self._loader.profile, profile)

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
        """
        Step the environment depending on the task
        """
        # Guard against stepping a terminated episode
        s = self._state
        if s.step_count >= s.max_steps or s.remaining_budget <= 0.0:
            raise ValueError(
                f"Episode already terminated at step {s.step_count} "
                f"(max_steps={s.max_steps}, remaining_budget={s.remaining_budget:.2f}). "
                f"Call reset() to start a new episode."
            )
        if self.task == "budget":
            return tasks.step_budget(s, action)
        elif self.task == "auction":
            return tasks.step_auction(s, action)
        elif self.task == "dynamic_campaign":
            return tasks.step_dynamic_campaign(s, action)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    # ---------------- STATE ----------------
    @property
    def state(self):
        s = self._state
        if self.task == "budget":
            return State(
            step_count= s.step_count,
            remaining_budget = s.remaining_budget,
            campaign_performance = s.conversion_rates.copy(),
            reward_buffer = list(s.reward_buffer),  # has to be JSON-serializable
            total_conversions = s.total_conversions,
            total_spend = s.total_spend,
            done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0
            )
        else:
            return State(
                step_count = s.step_count,
                remaining_budget = s.remaining_budget,
                competitor_bids = getattr(s, "competitor_bids", []),
                conversion_rates = getattr(s, "conversion_rates", []),
                total_conversions = getattr(s, "total_conversions", 0.0),
                total_spend = getattr(s, "total_spend", 0.0),
                done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0
            )


    # ---------------- MAX SCORE ----------------
    @property
    def max_possible_conversions(self) -> float:
        s = self._state
        return s.total_budget * max(getattr(s, "conversion_rates", [1.0]))