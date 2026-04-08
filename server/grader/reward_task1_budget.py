"""
reward_task1_budget.py
======================
Reward system for Task 1: budget allocation.

  Terminal trajectory bonus (0.25 weight):
    avg_conv_score   (0.70)
    avg_util_score   (0.20)
    avg_smooth_score (0.10)

Raw reward range:
  max: gate=1, all positive=1, no penalties → 1.0
  min: gate=0, max soft penalties           → -0.10
  → shift_scale maps [-0.10, 1.00] → [0, 1]
"""

from __future__ import annotations
import numpy as np
from .reward_base import (
    compute_conv_signal,
    compute_illegal_gate,
    compute_soft_penalties,
    compute_terminal_bonus,
    shift_scale,
    norm,
    W_SPEND,
    W_CARRYOVER,
)

# ---------------------------------------------------------------------------
# Task 1 positive signal weights — matched to grader
# ---------------------------------------------------------------------------
W_CONV        = 0.70
W_UTILIZATION = 0.20
W_SMOOTHNESS  = 0.10
RAW_MAX       = W_CONV + W_UTILIZATION + W_SMOOTHNESS   # 1.00

# Raw min: carryover + spend penalties
RAW_MIN_TASK1 = -(W_CARRYOVER + W_SPEND)                # -0.10


# ---------------------------------------------------------------------------
# Utilization pacing signal
# Linear — appropriate for task 1 since there are no market signals
# to suggest spending more at certain steps.
# Rewards the agent for keeping cumulative spend on track with ideal
# linear pacing through the episode.
# ---------------------------------------------------------------------------

def compute_utilization(
    total_spend:  float,
    total_budget: float,
) -> float:
    """
    Cumulative budget utilization progress.
    Consistent with grader's utilization_score formula.
    Peaks at 1.0 when total_spend == total_budget.
    Returns utilization_n in [0, 1].
    """
    utilization = total_spend / (total_budget + 1e-8)
    return float(max(0.0, min(1.0, 1.0 - abs(1.0 - utilization))))


# ---------------------------------------------------------------------------
# Smoothness signal
# ---------------------------------------------------------------------------

def compute_smoothness(
    spend:       float,
    total_spend: float,
    step_count:  int,
) -> float:
    """
    Per-step spend consistency vs running mean spend so far.
    Mirrors grader's spend_std/spend_mean at episode level.

    Returns 1.0 at step 0 (neutral, mean undefined).
    Returns smoothness_n in [0, 1].
    """
    if step_count <= 0:
        return 1.0
    spend_mean = total_spend / (step_count + 1e-8)
    deviation  = abs(spend - spend_mean) / (spend_mean + 1e-8)
    return float(max(0.0, min(1.0, 1.0 - deviation)))


# ---------------------------------------------------------------------------
# Core: compute task 1 normalized step reward
# ---------------------------------------------------------------------------

def compute_task1_step_reward(
    delayed_reward:    float,
    spend_penalty:     float,
    carryover_penalty: float,
    illegal_penalty:   float,
    spend:             float,
    total_spend:       float,
    total_budget:      float,
    step_count:        int,
    is_terminal:       bool  = False,
    cumulative_smooth: float = 0.0,
    cumulative_conv:   float = 0.0,
    max_possible_conv: float = 1.0,
) -> dict:
    """
    Compute normalized step reward for task 1.

    Parameters
    ----------
    delayed_reward    : raw conversion value (delayed one step)
    spend_penalty     : spend penalty from step function
    carryover_penalty : early aggression penalty
    illegal_penalty   : illegal allocation penalty
    spend             : budget spent this step
    total_spend       : cumulative spend including this step
    total_budget      : episode total budget
    step_count        : current step (before increment in step function)
    is_terminal       : True on last step
    cumulative_smooth : sum of per-step smoothness scores (for terminal bonus)
    cumulative_conv   : total_conversions so far (for terminal bonus)
    max_possible_conv : theoretical max conversions (for terminal bonus)

    Returns
    -------
    dict with step_reward and all component diagnostics
    """
    # --- Positive signals ---
    conv_n        = compute_conv_signal(delayed_reward)
    utilization_n = compute_utilization(total_spend, total_budget)
    smoothness_n  = compute_smoothness(spend, total_spend, step_count)

    positive = (
        W_CONV        * conv_n
      + W_UTILIZATION * utilization_n
      + W_SMOOTHNESS  * smoothness_n
    )

    # --- Illegal gate ---
    illegal_gate = compute_illegal_gate(illegal_penalty)

    # --- Soft penalties: carryover + spend ---
    spend_n, carryover_n, soft_penalty = compute_soft_penalties(
        spend_penalty, carryover_penalty
    )

    # --- Terminal bonus ---
    terminal_bonus = 0.0
    if is_terminal and step_count > 0:
        avg_conv_score   = float(min(1.0, cumulative_conv / (max_possible_conv + 1e-8)))
        avg_smooth_score = float(min(1.0, cumulative_smooth / step_count))
        avg_util_score   = compute_utilization(total_spend, total_budget)
        terminal_bonus   = compute_terminal_bonus({
            "conv":       (avg_conv_score,   0.70),
            "util":       (avg_util_score,   0.20),
            "smoothness": (avg_smooth_score, 0.10),
        })

    # --- Assemble with task 1 specific raw range ---
    gated       = illegal_gate * positive
    raw         = gated - soft_penalty
    step_reward = shift_scale(raw, RAW_MIN_TASK1, RAW_MAX)
    step_reward = float(max(0.0, min(1.0, step_reward + terminal_bonus)))

    return {
        "step_reward":           round(step_reward,    4),
	# Positive components
        "conv_component":        round(conv_n,          4),
        "utilization_component": round(utilization_n,   4),
        "smoothness_component":  round(smoothness_n,    4),
	# Gate and penalties
        "illegal_gate":          round(illegal_gate,    4),
        "illegal_component":     round(norm(illegal_penalty, 1.50), 4),
        "carryover_component":   round(carryover_n,     4),
        "spend_component":       round(spend_n,         4),
	# Terminal
        "terminal_bonus":        round(terminal_bonus,  4),
        "raw_combined":          round(raw,             4),
    }