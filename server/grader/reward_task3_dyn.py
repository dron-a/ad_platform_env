"""
reward_task3_dyn.py
=====================
Reward system for Task 3: Multi_Dynamics_Campaign.

Signals:
  Positive (weights sum to 1.0):
    conv_n            (0.40) : conversions this step / theoretical max
    bid_quality_n     (0.25) : CR-weighted win concentration on best campaigns
    pacing_n          (0.20) : opportunity-aware pacing (seasonal + events baked in)
    adaptability_n    (0.15) : response to conversion rate elevation above base

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
# Task 3 positive signal weights
# ---------------------------------------------------------------------------
W_CONV         = 0.40
W_BID          = 0.25
W_PACING       = 0.20
W_ADAPTABILITY = 0.15
RAW_MAX        = W_CONV + W_BID + W_PACING + W_ADAPTABILITY   # 1.00

# ---------------------------------------------------------------------------
# Opportunity-aware pacing signal
# Uses compiled conversion_rates (seasonal + events already baked in)
# and competitor_bids. No need to decompose — the environment already
# compiled these signals into the rates the agent observes.
# ---------------------------------------------------------------------------

def compute_opportunity_pacing(
    spend:                    float,
    remaining_budget:         float,
    step_count:               int,
    max_steps:                int,
    step_opportunity:         float,
    episode_mean_opportunity: float,
) -> float:
    """
    Opportunity-aware pacing for task 3.

    step_opportunity = sum(cr) / mean(competitor_bids)
      — naturally high when conversion rates are elevated (events, seasonality)
        and competitors are weak
      — already captures seasonal and event effects since cr is compiled

    ideal_fraction = (1/steps_remaining) * opp_ratio
    where opp_ratio = step_opportunity / episode_mean_opportunity
    clamped to [0.2, 3.0] to prevent extreme values

    pacing_error   = |actual_fraction - ideal_fraction| / (ideal_fraction + 1e-8)
    pacing_signal  = 1 - clamp(pacing_error, 0, 1)

    Returns pacing_signal in [0, 1].
    """
    steps_remaining = max_steps - step_count
    # FIXED
    if remaining_budget <= 0:
        return 1.0
    if steps_remaining <= 0:
        return float(max(0.0, min(1.0, spend / (remaining_budget + 1e-8))))

    opp_ratio = step_opportunity / (episode_mean_opportunity + 1e-8)
    opp_ratio = float(max(0.2, min(3.0, opp_ratio)))

    uniform_fraction = 1.0 / steps_remaining
    ideal_fraction   = float(min(1.0, uniform_fraction * opp_ratio))
    actual_fraction  = spend / (remaining_budget + 1e-8)

    pacing_error = abs(actual_fraction - ideal_fraction) / (ideal_fraction + 1e-8)
    return float(max(0.0, min(1.0, 1.0 - pacing_error)))


# ---------------------------------------------------------------------------
# Step opportunity score
# ---------------------------------------------------------------------------

def compute_step_opportunity(
    conversion_rates: list[float],
    competitor_bids:  list[float],
) -> float:
    """
    Raw opportunity score for this step.
    Uses compiled conversion_rates — seasonal and event effects already baked in.

    opportunity = sum(cr) / (mean(competitor_bids) + 1e-8)

    High when: conversion rates elevated (seasonal peak / market event)
               AND competitor bids are low (easier to win)
    Low when:  conversion rates depressed AND competitors are strong
    """
    total_cr  = sum(conversion_rates)
    mean_cb   = sum(competitor_bids) / (len(competitor_bids) + 1e-8)
    return float(total_cr / (mean_cb + 1e-8))


# ---------------------------------------------------------------------------
# Adaptability signal
# Measures agent response to conversion rate elevation above base.
# When market events or seasonality boost CRs, did the agent exploit it?
# ---------------------------------------------------------------------------

def compute_adaptability(
    conversion_rates:      list[float],
    base_conversion_rates: list[float],
    allocations:           list[float],
    bids:                  list[float],
    competitor_bids:       list[float],
) -> float:
    """
    Per-step adaptability signal for task 3.

    For each campaign:
      elevation   = cr / (base_cr + 1e-8) — how much above base is this step
      win_prob    = sigmoid(bid - competitor_bid)
      alloc_share = allocation / total_allocation

    If a campaign is elevated AND the agent won AND allocated to it:
      → high contribution → high adaptability score

    This teaches the agent to:
      1. Detect which campaigns are elevated this step (via campaign_performance obs)
      2. Increase bids and allocations on those campaigns
      3. Win auctions on elevated campaigns — this is what market event exploitation means

    Normalized against theoretical max where:
      - all campaigns are at 2x base (moderate event)
      - agent wins all auctions on best campaign with full allocation

    Returns adaptability_signal in [0, 1].
    """
    total_alloc = sum(allocations) + 1e-8

    weighted_exploit = 0.0
    for cr, base_cr, alloc, bid, cb in zip(
        conversion_rates, base_conversion_rates,
        allocations, bids, competitor_bids
    ):
        elevation   = cr / (base_cr + 1e-8)          # > 1 during events
        elevation   = float(max(0.0, elevation))
        win_prob    = 1.0 / (1.0 + np.exp(cb - bid))
        alloc_share = alloc / total_alloc
        weighted_exploit += elevation * win_prob * alloc_share

    # Theoretical max: elevation= calculated from the conversion rates
    max_elevation = max(
    cr / (base_cr + 1e-8)
    for cr, base_cr in zip(conversion_rates, base_conversion_rates)
    )
    ideal = max(max_elevation, 1.0) * 1.0 * 1.0
    return norm(weighted_exploit, ideal)


# ---------------------------------------------------------------------------
# Core: compute task 3 normalized step reward
# ---------------------------------------------------------------------------

def compute_task3_step_reward(
    delayed_reward:           float,
    bids:                     list[float],
    competitor_bids:          list[float],
    conversion_rates:         list[float],
    base_conversion_rates:    list[float],
    allocations:              list[float],
    spend:                    float,
    remaining_budget:         float,
    spend_penalty:            float,
    carryover_penalty:        float,
    illegal_penalty:          float,
    step_count:               int,
    max_steps:                int,
    step_opportunity:         float,
    episode_mean_opportunity: float,
    is_terminal:              bool  = False,
    cumulative_pacing:        float = 0.0,
    cumulative_bid:           float = 0.0,
    cumulative_conv:          float = 0.0,
    cumulative_adaptability:  float = 0.0,
    max_possible_conv:        float = 1.0,
) -> dict:
    """
    Compute normalized step reward for task 3.

    Returns dict with step_reward and all component diagnostics.
    """
    # --- Positive signals ---
    conv_n         = compute_conv_signal(delayed_reward)
    bid_n          = compute_bid_quality(
        bids, competitor_bids, conversion_rates, allocations
    )
    pacing_n       = compute_opportunity_pacing(
        spend, remaining_budget, step_count, max_steps,
        step_opportunity, episode_mean_opportunity,
    )
    adaptability_n = compute_adaptability(
        conversion_rates, base_conversion_rates,
        allocations, bids, competitor_bids,
    )

    positive = (
        W_CONV         * conv_n
      + W_BID          * bid_n
      + W_PACING       * pacing_n
      + W_ADAPTABILITY * adaptability_n
    )

    # --- Gate and soft penalties ---
    illegal_gate                     = compute_illegal_gate(illegal_penalty)
    spend_n, carryover_n, soft_penalty = compute_soft_penalties(
        spend_penalty, carryover_penalty
    )

    # --- Terminal bonus ---
    terminal_bonus = 0.0
    if is_terminal and step_count > 0:
        avg_conv_score     = float(min(1.0, cumulative_conv / (max_possible_conv + 1e-8)))
        avg_pacing_score   = float(min(1.0, cumulative_pacing      / step_count))
        avg_bid_quality    = float(min(1.0, cumulative_bid         / step_count))
        avg_adaptability   = float(min(1.0, cumulative_adaptability / step_count))
        terminal_bonus     = compute_terminal_bonus({
            "conv":          (avg_conv_score,   0.40),
            "bid":           (avg_bid_quality,  0.25),
            "pacing":        (avg_pacing_score, 0.20),
            "adaptability":  (avg_adaptability, 0.15),
        })

    # --- Assemble ---
    step_reward, raw = assemble_step_reward(
        positive_weighted = positive,
        illegal_gate      = illegal_gate,
        soft_penalty      = soft_penalty,
        raw_max           = RAW_MAX,
        terminal_bonus    = terminal_bonus,
    )

    return {
        "step_reward":              round(step_reward,      4),
        # Positive components
        "conv_component":           round(conv_n,            4),
        "bid_quality_component":    round(bid_n,             4),
        "pacing_component":         round(pacing_n,          4),
        "adaptability_component":   round(adaptability_n,    4),
        # Gate and penalties
        "illegal_gate":             round(illegal_gate,      4),
        "illegal_component":        round(norm(illegal_penalty, 1.50), 4),
        "carryover_component":      round(carryover_n,       4),
        "spend_component":          round(spend_n,           4),
        # Terminal
        "terminal_bonus":           round(terminal_bonus,    4),
        "raw_combined":             round(raw,               4),
    }