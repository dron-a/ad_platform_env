"""
reward_task2_auction.py
=======================
Reward system for Task 2: auction.

Signals:
  Positive (weights sum to 1.0):
    conv_n          (0.50) : conversions this step / theoretical max
    bid_quality_n   (0.30) : CR-weighted win concentration on best campaigns
    pacing_n        (0.20) : competitor-aware budget pacing

Raw reward range:
  max: gate=1, all positive=1, no penalties → 1.0
  min: gate=0, max soft penalties           → -0.10
  → shift_scale maps [-0.10, 1.00] → [0, 1]
"""

from __future__ import annotations
import numpy as np
from .reward_base import (
    compute_conv_signal,
    compute_bid_quality,
    compute_illegal_gate,
    compute_soft_penalties,
    compute_terminal_bonus,
    assemble_step_reward,
    norm,
)

# ---------------------------------------------------------------------------
# Task 2 positive signal weights
# ---------------------------------------------------------------------------
W_CONV    = 0.50
W_BID     = 0.30
W_PACING  = 0.20
RAW_MAX   = W_CONV + W_BID + W_PACING   # 1.00

# ---------------------------------------------------------------------------
# Competitor-aware pacing signal
# No seasonality, no market events.
# Rewards spending proportionally to how winnable the current auctions are.
# When aggregate win probability is high → spend more is justified.
# When competitors are strong → conserve budget for better steps.
# ---------------------------------------------------------------------------

def compute_competitor_aware_pacing(
    bids:             list[float],
    competitor_bids:  list[float],
    conversion_rates: list[float],
    allocations:      list[float],
    spend:            float,
    remaining_budget: float,
    step_count:       int,
    max_steps:        int,
) -> float:
    """
    Competitor-aware pacing signal for task 2.

    Ideal spend fraction this step:
      Weighted by aggregate win probability across campaigns,
      relative to a uniform baseline of 1/steps_remaining.

    win_quality = sum(cr * win_prob) / sum(cr)   — in [0,1]
    When win_quality is high → agent should spend more than uniform.
    When win_quality is low  → agent should conserve.

    ideal_fraction = (1/steps_remaining) * (0.5 + win_quality)
      Factor (0.5 + win_quality) ranges [0.5, 1.5]:
        win_quality=0 → spend at half the uniform rate (conserve)
        win_quality=1 → spend at 1.5x uniform rate (exploit)

    pacing_error   = |actual_fraction - ideal_fraction| / ideal_fraction
    pacing_signal  = 1 - clamp(pacing_error, 0, 1)

    Returns pacing_signal in [0, 1].
    """
    steps_remaining = max_steps - step_count
    # FIXED
    if remaining_budget <= 0:
        return 1.0
    if steps_remaining <= 0:
        # Last step — reward spending all remaining budget
        return float(max(0.0, min(1.0, spend / (remaining_budget + 1e-8))))

    # Aggregate win quality: CR-weighted win probability
    total_cr    = sum(conversion_rates) + 1e-8
    win_quality = sum(
        cr * (1.0 / (1.0 + np.exp(cb - bid)))
        for bid, cb, cr in zip(bids, competitor_bids, conversion_rates)
    ) / total_cr
    win_quality = float(max(0.0, min(1.0, win_quality)))

    # Ideal spend fraction
    uniform_fraction = 1.0 / steps_remaining
    ideal_fraction   = uniform_fraction * (0.5 + win_quality)
    ideal_fraction   = float(min(ideal_fraction, 1.0))

    # Actual fraction of remaining budget spent
    actual_fraction = spend / (remaining_budget + 1e-8)

    # Relative pacing error
    pacing_error = abs(actual_fraction - ideal_fraction) / (ideal_fraction + 1e-8)
    return float(max(0.0, min(1.0, 1.0 - pacing_error)))


# ---------------------------------------------------------------------------
# Core: compute task 2 normalized step reward
# ---------------------------------------------------------------------------

def compute_task2_step_reward(
    delayed_reward:    float,
    bids:              list[float],
    competitor_bids:   list[float],
    conversion_rates:  list[float],
    allocations:       list[float],
    spend:             float,
    remaining_budget:  float,
    spend_penalty:           float,
    carryover_penalty: float,
    illegal_penalty:   float,
    step_count:        int,
    max_steps:         int,
    is_terminal:       bool  = False,
    cumulative_pacing: float = 0.0,
    cumulative_bid:    float = 0.0,
    cumulative_conv:   float = 0.0,
    max_possible_conv: float = 1.0,
) -> dict:
    """
    Compute normalized step reward for task 2.

    Returns dict with step_reward and all component diagnostics.
    """
    # --- Positive signals ---
    conv_n      = compute_conv_signal(delayed_reward)
    bid_n       = compute_bid_quality(
        bids, competitor_bids, conversion_rates, allocations
    )
    pacing_n    = compute_competitor_aware_pacing(
        bids, competitor_bids, conversion_rates, allocations,
        spend, remaining_budget, step_count, max_steps
    )

    positive = W_CONV * conv_n + W_BID * bid_n + W_PACING * pacing_n

    # --- Gate and soft penalties ---
    illegal_gate                     = compute_illegal_gate(illegal_penalty)
    spend_n, carryover_n, soft_penalty = compute_soft_penalties(
        spend_penalty, carryover_penalty
    )

    # --- Terminal bonus ---
    terminal_bonus = 0.0
    if is_terminal and step_count > 0:
        avg_conv_score = float(min(1.0, cumulative_conv / (max_possible_conv + 1e-8)))
        avg_pacing_score = float(min(1.0, cumulative_pacing / step_count))
        avg_bid_quality = float(min(1.0, cumulative_bid    / step_count))
        terminal_bonus = compute_terminal_bonus({
            "conv":   (avg_conv_score,   0.50),
            "bid":    (avg_bid_quality,  0.30),
            "pacing": (avg_pacing_score, 0.20),
        })

    # --- Assemble ---
    step_reward, raw = assemble_step_reward(
        positive_weighted = positive,
        illegal_gate = illegal_gate,
        soft_penalty = soft_penalty,
        raw_max = RAW_MAX,
        terminal_bonus = terminal_bonus,
    )

    return {
        "step_reward":          round(step_reward,    4),
        # Positive components
        "conv_component":       round(conv_n,          4),
        "bid_quality_component":round(bid_n,           4),
        "pacing_component":     round(pacing_n,        4),
        # Gate and penalties
        "illegal_gate":         round(illegal_gate,    4),
        "illegal_component":    round(norm(illegal_penalty, 1.50), 4),
        "carryover_component":  round(carryover_n,     4),
        "spend_component":      round(spend_n,         4),
        # Terminal
        "terminal_bonus":       round(terminal_bonus,  4),
        "raw_combined":         round(raw,             4),
    }