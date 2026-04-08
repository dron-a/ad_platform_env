"""
reward_base.py
==============
Shared constants, helpers, and core reward components used by all three
task-specific reward modules.

    reward_task1_budget.py   — imports this
    reward_task2_auction.py  — imports this
    reward_task3_multi.py    — imports this

Nothing in this file is task-specific. All task-specific signal computation
lives in the respective task module.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Theoretical bounds
# Derived from AdPlatformState defaults
# ---------------------------------------------------------------------------

MAX_CONV_PER_STEP: float = 300.0 * 0.05 * 1.10   # ≈ 16.5
MAX_SPEND_PENALTY: float = 0.09    # penalty_alpha=1.0, beta=2.0, frac=0.30
MAX_CARRYOVER_PENALTY: float = 0.20  # 0.2 * 1.0^2 * (1-0) at step 0
MAX_ILLEGAL_PENALTY: float = 1.50   # 3 campaigns * 0.5 per negative alloc

# ---------------------------------------------------------------------------
# Shared penalty weights — identical across all tasks
# ---------------------------------------------------------------------------
W_CARRYOVER: float = 0.07
W_SPEND:     float = 0.03

# ---------------------------------------------------------------------------
# Helper: clamp-normalize value to [0, 1] against a known ceiling
# ---------------------------------------------------------------------------

def norm(value: float, ceiling: float) -> float:
    """Normalize value to [0,1] against theoretical ceiling."""
    if ceiling <= 0:
        return 0.0
    return float(max(0.0, min(1.0, value / ceiling)))


# ---------------------------------------------------------------------------
# Illegal gate
# Multiplicative — suppresses entire positive reward when illegal actions
# occur. Cannot be compensated by good performance elsewhere.
# ---------------------------------------------------------------------------

def compute_illegal_gate(illegal_penalty: float) -> float:
    """
    Returns a gate in [0, 1].
    0.0 when illegal_penalty is at maximum — fully suppresses positive reward.
    1.0 when no illegal actions — positive reward passes through unchanged.
    """
    illegal_n = norm(illegal_penalty, MAX_ILLEGAL_PENALTY)
    return float(1.0 - illegal_n)


# ---------------------------------------------------------------------------
# Soft penalties — additive, shared across all tasks
# ---------------------------------------------------------------------------

def compute_soft_penalties(
    spend_penalty:          float,
    carryover_penalty: float,
) -> tuple[float, float, float]:
    """
    Normalize spend and carryover penalties.
    Returns (spend_n, carryover_n, total_soft_penalty_contribution).
    total = W_SPEND * spend_n + W_CARRYOVER * carryover_n
    """
    spend_n = norm(spend_penalty, MAX_SPEND_PENALTY)
    carryover_n = norm(carryover_penalty, MAX_CARRYOVER_PENALTY)
    total = W_SPEND * spend_n + W_CARRYOVER * carryover_n
    return spend_n, carryover_n, total


# ---------------------------------------------------------------------------
# Conversion signal — shared across all tasks
# ---------------------------------------------------------------------------

def compute_conv_signal(delayed_reward: float) -> float:
    """Normalize per-step conversion value to [0, 1]."""
    return norm(delayed_reward, MAX_CONV_PER_STEP)


# ---------------------------------------------------------------------------
# Bid quality signal — shared by task 2 and task 3
# Not used by task 1 (no bidding).
#
# Rewards strategic concentration of auction wins on high-CR campaigns,
# weighted by allocation share so the agent is rewarded for putting more
# money behind campaigns it is winning on the best terms.
# ---------------------------------------------------------------------------

def compute_bid_quality(
    bids:             list[float],
    competitor_bids:  list[float],
    conversion_rates: list[float],
    allocations:      list[float],
) -> float:
    """
    Returns bid_quality in [0, 1].

    For each campaign:
      win_prob    = sigmoid(bid - competitor_bid)
      alloc_share = allocation / total_allocation
      contribution = cr * win_prob * alloc_share

    Normalized against theoretical max where win_prob=1 on best campaign
    with full allocation share.
    """
    total_alloc = sum(allocations) + 1e-8
    best_cr = max(conversion_rates, default=1.0)

    weighted_win = 0.0
    for bid, cb, cr, alloc in zip(bids, competitor_bids, conversion_rates, allocations):
        win_prob = 1.0 / (1.0 + np.exp(cb - bid))
        alloc_share = alloc / total_alloc
        weighted_win += cr * win_prob * alloc_share

    ideal = best_cr * 1.0 * 1.0   # win_prob=1, full alloc share, best campaign
    return norm(weighted_win, ideal)


# ---------------------------------------------------------------------------
# Final normalization: shift raw combined value to [0, 1]
# ---------------------------------------------------------------------------

def shift_scale(raw: float, raw_min: float, raw_max: float) -> float:
    """
    Linearly map raw from [raw_min, raw_max] to [0, 1].
    Clamps output to [0, 1] for safety.
    """
    span = raw_max - raw_min
    if span <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (raw - raw_min) / span)))


# ---------------------------------------------------------------------------
# Terminal trajectory bonus — shared structure across all tasks
# Each task passes its own averaged signal dict.
#
# terminal_bonus = TERMINAL_WEIGHT * weighted_sum(averaged_signals)
# Clamped to [0, 1] before adding to step_reward.
# ---------------------------------------------------------------------------

TERMINAL_WEIGHT: float = 0.25

def compute_terminal_bonus(averaged_signals: dict[str, tuple[float, float]]) -> float:
    """
    Compute terminal trajectory bonus from episode-averaged step signals.

    averaged_signals: dict of { signal_name: (avg_value, weight) }
    Weights should sum to 1.0.

    Example for task 2:
        {
            "conv":    (avg_conv_score,    0.50),
            "pacing":  (avg_pacing_score,  0.30),
            "bid":     (avg_bid_quality,   0.20),
        }

    Returns terminal_bonus in [0, TERMINAL_WEIGHT].
    """
    weighted_sum = sum(
        float(max(0.0, min(1.0, val))) * weight
        for val, weight in averaged_signals.values()
    )
    return float(TERMINAL_WEIGHT * max(0.0, min(1.0, weighted_sum)))


# ---------------------------------------------------------------------------
# Assemble final step reward from components
# Shared assembly logic — task modules call this after computing their
# own positive signals.
# ---------------------------------------------------------------------------

def assemble_step_reward(
    positive_weighted: float,   # already weighted sum of positive signals
    illegal_gate: float,   # from compute_illegal_gate()
    soft_penalty: float,   # from compute_soft_penalties()[2]
    raw_max: float,   # theoretical max of positive_weighted
    terminal_bonus: float = 0.0,
) -> tuple[float, float]:
    """
    Apply gate, subtract soft penalties, normalize to [0,1], add terminal bonus.

    raw_min = -(W_CARRYOVER + W_SPEND) = -0.10  (always, since soft penalties shared)
    raw_max = passed in by task module (sum of positive weights)

    Returns (step_reward, raw_combined).
    """
    raw_min = -(W_CARRYOVER + W_SPEND)   # -0.10

    gated = illegal_gate * positive_weighted
    raw = gated - soft_penalty

    step_reward = shift_scale(raw, raw_min, raw_max)
    step_reward = float(max(0.0, min(1.0, step_reward + terminal_bonus)))

    return step_reward, raw
