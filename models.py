# models.py

from collections import deque
import numpy as np
from pydantic import Field
from typing import List, Dict, Any, Optional
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# CampaignProfile — inject real historical data at reset time
# ---------------------------------------------------------------------------

class CampaignProfile(dict):
    """
    A real user's historical campaign data, passed to reset() to replace
    synthetic defaults. All fields are optional — any omitted field falls
    back to the environment's built-in defaults.

    Example (from a real Google Ads export):
        profile = CampaignProfile(
            conversion_rates=[0.04, 0.02, 0.015],        # historical CVR per campaign
            competitor_bids=[0.80, 0.60, 0.45],           # average CPC competitors paid
            bid_volatility=[0.08, 0.12, 0.06],            # std-dev of competitor bid swings
            seasonal_multipliers=[1.3, 1.0, 0.8, 1.2, ...],  # one per step (e.g. day-of-week)
            market_events={5: [1.4, 1.0, 0.9], 18: [1.0, 1.5, 1.0]},  # known sale events
            total_budget=5000.0,
        )
        env.reset(task="auction", profile=profile)

    Fields:
        conversion_rates       list[float]  — CVR per campaign (conversions per $ spent)
        competitor_bids        list[float]  — baseline competitor bid per campaign
        bid_volatility         list[float]  — per-campaign std-dev of competitor bid noise
                                             (replaces the fixed ±10% uniform noise)
        seasonal_multipliers   list[float]  — one multiplier per step; if shorter than
                                             max_steps the list wraps around
        market_events          dict[int, list[float]]  — step → per-campaign CVR multiplier
        total_budget           float        — episode budget (overrides state default)
    """

    VALID_KEYS = {
        "conversion_rates",
        "competitor_bids",
        "bid_volatility",
        "seasonal_multipliers",
        "market_events",
        "total_budget",
    }

    def __init__(self, **kwargs):
        # --- Unknown key check ---
        unknown = set(kwargs) - self.VALID_KEYS
        if unknown:
            raise ValueError(
                f"CampaignProfile received unknown fields: {unknown}. "
                f"Valid fields are: {self.VALID_KEYS}"
            )

        # --- conversion_rates ---
        if "conversion_rates" in kwargs:
            val = kwargs["conversion_rates"]
            if not isinstance(val, (list, tuple)):
                raise TypeError(f"conversion_rates must be a list, got {type(val)}")
            if len(val) == 0:
                raise ValueError("conversion_rates cannot be empty")
            if not all(isinstance(v, (int, float)) and v >= 0 for v in val):
                raise ValueError("conversion_rates must contain non-negative numbers")

        # --- competitor_bids ---
        if "competitor_bids" in kwargs:
            val = kwargs["competitor_bids"]
            if not isinstance(val, (list, tuple)):
                raise TypeError(f"competitor_bids must be a list, got {type(val)}")
            if len(val) == 0:
                raise ValueError("competitor_bids cannot be empty")
            if not all(isinstance(v, (int, float)) and v > 0 for v in val):
                raise ValueError("competitor_bids must contain positive numbers")

        # --- bid_volatility ---
        if "bid_volatility" in kwargs:
            val = kwargs["bid_volatility"]
            if not isinstance(val, (list, tuple)):
                raise TypeError(f"bid_volatility must be a list, got {type(val)}")
            if len(val) == 0:
                raise ValueError("bid_volatility cannot be empty")
            if not all(isinstance(v, (int, float)) and v >= 0 for v in val):
                raise ValueError("bid_volatility must contain non-negative numbers")

        # --- seasonal_multipliers ---
        if "seasonal_multipliers" in kwargs:
            val = kwargs["seasonal_multipliers"]
            if not isinstance(val, (list, tuple)):
                raise TypeError(f"seasonal_multipliers must be a list, got {type(val)}")
            if len(val) == 0:
                raise ValueError("seasonal_multipliers cannot be empty")
            if not all(isinstance(v, (int, float)) and v >= 0 for v in val):
                raise ValueError("seasonal_multipliers must contain non-negative numbers")

        # --- market_events ---
        if "market_events" in kwargs:
            val = kwargs["market_events"]
            if not isinstance(val, dict):
                raise TypeError(f"market_events must be a dict, got {type(val)}")
            for step, multipliers in val.items():
                if not isinstance(step, int) or step < 0:
                    raise ValueError(
                        f"market_events keys must be non-negative integers, got {step}"
                    )
                if not isinstance(multipliers, (list, tuple)):
                    raise TypeError(
                        f"market_events values must be lists, got {type(multipliers)} at step {step}"
                    )
                if not all(isinstance(m, (int, float)) and m >= 0 for m in multipliers):
                    raise ValueError(
                        f"market_events multipliers must be non-negative numbers at step {step}"
                    )

        # --- total_budget ---
        if "total_budget" in kwargs:
            val = kwargs["total_budget"]
            if not isinstance(val, (int, float)):
                raise TypeError(f"total_budget must be a number, got {type(val)}")
            if val <= 0:
                raise ValueError(f"total_budget must be positive, got {val}")

        # --- Cross-field shape consistency ---
        # Per-campaign fields must all have the same length if more than one is provided
        campaign_fields = {
            k: kwargs[k] for k in ("conversion_rates", "competitor_bids", "bid_volatility")
            if k in kwargs
        }
        if len(campaign_fields) > 1:
            lengths = {k: len(v) for k, v in campaign_fields.items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"Per-campaign fields must all have the same length. "
                    f"Got: {lengths}"
                )

        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Action / Observation / State
# ---------------------------------------------------------------------------

class AdPlatformAction(Action):
    allocations: list[float] = Field(..., description="Budget allocation per campaign")
    bids: list[float] = Field(
        default_factory=list,
        description="Bid per campaign (required for auction/dynamic_campaign tasks, ignored for budget task)"
    )


class AdPlatformObservation(Observation):
    """Observation returned to the agent at each step."""
    step: int = Field(default=0, description="Current timestep in the episode")
    total_budget: float = Field(default=1000.0, description="Total episode budget")
    remaining_budget: float = Field(default=0.0, description="Remaining budget for the episode")
    campaign_performance: list[float] = Field(
        default_factory=list,
        description="Conversion rate per campaign"
    )
    competitor_bids: list[float] = Field(
        default_factory=list,
        description="Current competitor bid per campaign (auction/dynamic_campaign tasks)"
    )
    obs_history: list[dict] = Field(
        default_factory=list,
        description=(
            "Last K steps of context: each entry has keys "
            "{step, spend, conversions, competitor_bids, allocations, bids}. "
            "Allows the agent to observe trends and competitor patterns over time."
        )
    )

    # Step-level reward diagnostics
    reward_breakdown: Dict[str, float] | None = Field(default=None)

    # Terminal-step grader scores
    grader_final_score: float | None = Field(default=None)
    grader_conversion_score: float | None = Field(default=None)
    grader_utilization_score: float | None = Field(default=None)
    grader_bid_efficiency: float | None = Field(default=None)
    grader_total_conversions: float | None = Field(default=None)
    grader_total_spend: float | None = Field(default=None)
    grader_budget_remaining: float | None = Field(default=None)
    grader_max_possible_conversions: float | None = Field(default=None)

    # Previous episode grader scores (set at reset, available from step 0)
    prev_conversion_score: float = Field(default=0.0)
    prev_utilization_score: float = Field(default=0.0)
    prev_bid_efficiency: float = Field(default=0.0)
    prev_final_score: float = Field(default=0.0)
    prev_episode_graded: bool = Field(default=False)

    reward: float = Field(default=0.0)
    done: bool = Field(default=False)


class AdPlatformState(State):
    step_count: int = Field(default=0, description="Internal step counter")
    total_budget: float = Field(default=1000.0, description="Total budget")
    remaining_budget: float = Field(default=1000.0, description="Remaining budget")

    base_conversion_rates: List[float] = Field(default_factory=lambda: [0.05, 0.03, 0.02])
    conversion_rates: List[float] = Field(default_factory=list)

    base_competitor_bids: List[float] = Field(default_factory=lambda: [0.5, 0.4, 0.3])
    # Per-campaign bid volatility (std-dev fraction). Default replicates old ±10% uniform noise.
    bid_volatility: List[float] = Field(default_factory=lambda: [0.06, 0.06, 0.06])
    competitor_bids: List[float] = Field(default_factory=list)
    prev_agent_bids: List[float] = Field(default_factory=list)

    # Per-step seasonal multipliers (wraps if shorter than max_steps)
    seasonal_multipliers: List[float] = Field(default_factory=list)

    total_conversions: float = Field(default=0.0, description="Total conversions over episode")
    total_spend: float = Field(default=0.0, description="Total budget spent")
    spend_history: List[float] = Field(default_factory=list, description="Spend per step")

    # Rolling observation history — last `history_window` steps surfaced to the agent
    obs_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rolling history of step records included in observations"
    )
    history_window: int = Field(
        default=5,
        description="Number of past steps to include in each observation"
    )

    reward_buffer: deque = Field(
        default_factory=lambda: deque(maxlen=1),
        description="Buffer for delayed reward"
    )

    max_steps: int = Field(default=30, description="Max steps per episode")
    max_fraction_per_step: float = Field(
        default=0.3,
        description="Maximum fraction of budget that can be spent in one step"
    )

    penalty_alpha: float = Field(default=1.0, description="Spend penalty coefficient")
    penalty_beta: float = Field(default=2.0, description="Exponent for spend penalty")

    seed: int = 42
    alpha: float = 0.3  # competitor responsiveness
    realism_mode: str = "fixed"

    seasonal_amplitude: float = 0.1
    seasonal_period: int = 24

    market_events: dict = Field(default_factory=dict)

    # ---- Grader persistence across episodes ----
    prev_conversion_score: float = Field(default=0.0)
    prev_utilization_score: float = Field(default=0.0)
    prev_bid_efficiency: float = Field(default=0.0)
    prev_final_score: float = Field(default=0.0)
    prev_episode_graded: bool = Field(default=False)

    # ---------------- cumulative variables for episodic indications --------------------
    cumulative_bid_quality: float = Field(default=0.0)
    cumulative_pacing_score: float = Field(default=0.0)

    # -----------------last step reward ----------------------------
    last_step_reward: float = Field(default=0.0)

    # -------------------- reward Bonus for dynamic rubric calculation -------------------------
    reward_bounds: dict = Field(default_factory=dict)




    def __init__(self, realism_mode="fixed", seed=42, **data):
        super().__init__(**data)
        self.seed = seed
        self.realism_mode = realism_mode
        self.base_conversion_rates = [0.05, 0.03, 0.02]

    def apply_profile(self, profile: Optional["CampaignProfile"]) -> None:
        """
        Apply a resolved CampaignProfile to state, overriding synthetic defaults.

        Called at the top of every reset(). Works with profiles from three sources:
        - YAML file loaded via ProfileLoader
        - Runtime CampaignProfile passed directly to reset()
        - Merged profile from ProfileLoader.resolve(yaml_profile, runtime_profile)
        - Only fields explicitly present in the profile are applied
        - Anything absent in the profile stays as the synthetic default
        """
        if not profile:
            return

        n_campaigns = len(self.base_conversion_rates)

        # --- Shape validation before any state mutation ---
        # Check all per-campaign fields that are present in the profile
        per_campaign_fields = {
            k: profile[k] for k in ("conversion_rates", "competitor_bids", "bid_volatility")
            if k in profile
        }
        for field_name, values in per_campaign_fields.items():
            if len(values) != n_campaigns:
                raise ValueError(
                    f"Profile field '{field_name}' has {len(values)} campaigns "
                    f"but environment expects {n_campaigns}. "
                    f"All per-campaign fields must match the environment campaign count."
                )

        # --- Apply per-campaign fields ---
        if "conversion_rates" in profile:
            self.base_conversion_rates = list(profile["conversion_rates"])

        if "competitor_bids" in profile:
            self.base_competitor_bids = list(profile["competitor_bids"])

        if "bid_volatility" in profile:
            self.bid_volatility = list(profile["bid_volatility"])

        # --- Apply episode-level fields ---
        if "seasonal_multipliers" in profile:
            self.seasonal_multipliers = list(profile["seasonal_multipliers"])

        if "market_events" in profile:
            self.market_events = {
                int(k): list(v) for k, v in profile["market_events"].items()
            }

        if "total_budget" in profile:
            self.total_budget = float(profile["total_budget"])
            # remaining_budget will be set to total_budget in reset body
            # after apply_profile returns — no need to set it here


    def set_conversion_rates(self) -> None:
        np.random.seed(self.seed)
        if self.realism_mode == "fixed":
            self.conversion_rates = [r * 1.0 for r in self.base_conversion_rates]
        elif self.realism_mode == "realistic":
            self.conversion_rates = [
                r * (1 + 0.01 * ((i + self.seed) % 3))
                for i, r in enumerate(self.base_conversion_rates)
            ]
        else:
            raise ValueError(f"Unknown realism_mode: {self.realism_mode}")

    def get_seasonal_multiplier(self) -> float:
        """
        Return the seasonal multiplier for the current step.
        Uses profile-supplied per-step multipliers if available,
        otherwise falls back to the sine-wave default.
        """
        if self.seasonal_multipliers:
            return self.seasonal_multipliers[self.step_count % len(self.seasonal_multipliers)]
        return 1 + self.seasonal_amplitude * np.sin(
            2 * np.pi * self.step_count / self.seasonal_period
        )

    def sample_competitor_bid(self, campaign_idx: int) -> float:
        """
        Sample a competitor bid for one campaign using historically-grounded volatility
        instead of flat uniform noise. Uses a seeded normal distribution parameterised
        by bid_volatility, then applies adaptive response to previous agent bids.
        """
        base_bid = self.base_competitor_bids[campaign_idx]
        vol = self.bid_volatility[campaign_idx]
        rng = np.random.default_rng(self.seed + self.step_count + campaign_idx)
        # Normal noise scaled by volatility — more realistic than uniform ±10%
        noise_factor = 1 + rng.normal(0, vol)
        noisy_base = base_bid * max(0.5, noise_factor)   # floor at 50% of base
        # Adaptive response to previous agent bid
        prev_ab = self.prev_agent_bids[campaign_idx] if self.prev_agent_bids else 0.0
        agent_effect = 1 + self.alpha * (prev_ab / (base_bid + 1e-8) - 1)
        return max(0.01, noisy_base * agent_effect)

    def record_step(self, spend: float, conversions: float,
                    allocations: list, bids: list) -> None:
        """Append a step record to obs_history, keeping only the last history_window entries."""
        record = {
            "step": self.step_count,
            "spend": round(spend, 4),
            "conversions": round(conversions, 6),
            "competitor_bids": [round(b, 4) for b in self.competitor_bids],
            "allocations": [round(a, 4) for a in allocations],
            "bids": [round(b, 4) for b in bids],
        }
        self.obs_history.append(record)
        if len(self.obs_history) > self.history_window:
            self.obs_history.pop(0)
