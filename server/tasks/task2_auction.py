# server/auction_environment.py

import numpy as np
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState
from ..grader import *

# ---------------- RESET ----------------
def reset(state: AdPlatformState, profile: dict | None = None):
    """
    Reset environment for new episode.

    Before wiping:
      - Runs grader on completed episode
      - Persists grader scores to state for next episode's observation

    After wiping:
      - Resets all accumulator variables
      - Returns observation with previous episode grader scores so the agent can condition
         its new-episode policy on its prior performance.
    """
    s = state

    # --- Persist grader scores from completed episode ---
    if s.step_count > 0:
        grader_result            = compute_auction_score(s)
        s.prev_conversion_score  = grader_result["conversion_score"]
        s.prev_utilization_score = grader_result["utilization_score"]
        s.prev_bid_efficiency    = grader_result["bid_efficiency"]
        s.prev_final_score       = grader_result["final_score"]
        s.prev_episode_graded    = True

    # --- Apply campaign profile (overrides defaults with real data) and wipe episode state ---
    s.apply_profile(profile)
    s.step_count               = 0
    s.remaining_budget         = s.total_budget        # fixed original typo
    s.total_conversions        = 0.0
    s.total_spend              = 0.0
    s.obs_history.clear()
    s.reward_buffer.clear()
    s.conversion_rates         = s.base_conversion_rates.copy()
    s.prev_agent_bids          = [0.0] * len(s.conversion_rates)
    s.competitor_bids          = s.base_competitor_bids.copy()
    s.spend_history.clear()

    # --- Reset task 2 accumulators ---
    s.cumulative_bid_quality   = 0.0
    s.cumulative_pacing_score  = 0.0

    return AdPlatformObservation(
        step                 = s.step_count,
        total_budget         = s.total_budget,
        remaining_budget     = s.remaining_budget,
        campaign_performance = s.conversion_rates,
        competitor_bids      = s.competitor_bids,
        obs_history          = [],
        # Previous episode grader scores for policy conditioning
        prev_conversion_score  = s.prev_conversion_score,
        prev_utilization_score = s.prev_utilization_score,
        prev_bid_efficiency    = s.prev_bid_efficiency,
        prev_final_score       = s.prev_final_score,
        prev_episode_graded    = s.prev_episode_graded,
    )


def step(state: AdPlatformState, action: AdPlatformAction) -> AdPlatformObservation:
    s = state

    allocations = action.allocations
    bids = action.bids

    assert len(bids) == len(s.conversion_rates), "Bid length mismatch"
    assert len(allocations) == len(s.conversion_rates), "Allocation length mismatch"

    # ----------------------------
    # Generate competitor bids using historically-grounded volatility
    # ----------------------------
    s.competitor_bids = [
        s.sample_competitor_bid(i) for i in range(len(s.base_competitor_bids))
    ]

    # ----------------------------
    # Determine pacing limit
    # ----------------------------
    if s.step_count == s.max_steps - 1:
        pacing_limit = s.remaining_budget
    else:
        pacing_limit = s.max_fraction_per_step * s.remaining_budget

    # ----------------------------
    # Illegal allocation penalty
    # ----------------------------
    illegal_penalty = 0.0
    for a in allocations:
        if a < 0:
            illegal_penalty += 0.5
        elif a > pacing_limit:
            illegal_penalty += 0.2

    # ----------------------------
    # Clamp allocations
    # ----------------------------
    allocations = [max(0.0, min(a, pacing_limit)) for a in allocations]

    # ----------------------------
    # Budget spend
    # ----------------------------
    spend = min(sum(allocations), s.remaining_budget)
    total_available = s.remaining_budget
    s.remaining_budget -= spend
    s.total_spend += spend

    spend_ratio = spend / (total_available + 1e-8) if total_available > 0 else 0.0

    # ----------------------------
    # Conversions (smooth win probability + seasonal)
    # ----------------------------
    seasonal = s.get_seasonal_multiplier()
    conversions = 0.0
    for a, bid, cb, cr in zip(allocations, bids, s.competitor_bids, s.conversion_rates):
        win_prob     = 1.0 / (1.0 + np.exp(cb - bid)) # sigmoid
        conversions += a * cr * seasonal * win_prob
    s.total_conversions += conversions

    # --- Record step history ---
    s.record_step(
        spend=spend, conversions=conversions,
        allocations=allocations, bids=bids
    )

    # ----------------------------
    # --- Delayed reward ---
    # ----------------------------
    s.reward_buffer.append(conversions)
    if len(s.reward_buffer) == s.reward_buffer.maxlen:
        delayed_reward = s.reward_buffer.popleft()
    else:
        delayed_reward = 0.0

    # ----------------------------
    # Spend penalty
    # ----------------------------
    frac = spend / (s.total_budget + 1e-8)
    spend_penalty = s.penalty_alpha * (frac ** s.penalty_beta)

    # ----------------------------
    # Carryover penalty (early aggressive spending)
    # ----------------------------
    if s.step_count < s.max_steps - 1:
        time_factor = s.step_count / s.max_steps
        carryover_penalty = 0.2 * (spend_ratio ** 2) * (1 - time_factor)
    else:
        carryover_penalty = 0.0

    # --- Accumulate bid quality and pacing ---
    bid_quality = compute_bid_quality(
        bids, s.competitor_bids, s.conversion_rates, allocations
    )
    pacing_signal = compute_competitor_aware_pacing(
        bids, s.competitor_bids, s.conversion_rates, allocations,
        spend, total_available, s.step_count, s.max_steps
    )
    s.cumulative_bid_quality  += bid_quality
    s.cumulative_pacing_score += pacing_signal

    # --- Terminal check ---
    s.prev_agent_bids = bids.copy()
    s.step_count += 1
    done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

    # --- Terminal: grader for observation only ---
    grader_result = {}
    if done:
        grader_result = compute_auction_score(s)

    # --- Normalized step reward ---
    max_possible = s.total_budget * max(s.conversion_rates, default=1.0)
    reward_breakdown = compute_task2_step_reward(
        delayed_reward    = delayed_reward,
        bids              = bids,
        competitor_bids   = s.competitor_bids,
        conversion_rates  = s.conversion_rates,
        allocations       = allocations,
        spend             = spend,
        remaining_budget  = total_available,
        spend_penalty     = spend_penalty,
        carryover_penalty = carryover_penalty,
        illegal_penalty   = illegal_penalty,
        step_count        = s.step_count,
        max_steps         = s.max_steps,
        is_terminal       = done,
        cumulative_pacing = s.cumulative_pacing_score,
        cumulative_bid    = s.cumulative_bid_quality,
        cumulative_conv   = s.total_conversions,
        max_possible_conv = max_possible,
    )
    reward = reward_breakdown["step_reward"]
    s.last_step_reward = reward

    # --- Build observation ---
    # Grader fields default to None mid-episode so the agent knows they are
    # not yet meaningful (avoids treating 0.0 as a real score).
    return AdPlatformObservation(
        step                 = s.step_count,
        total_budget         = s.total_budget,
        remaining_budget     = s.remaining_budget,
        campaign_performance = s.conversion_rates,
        competitor_bids      = s.competitor_bids,
        obs_history          = list(s.obs_history),
        reward               = reward,
        done                 = done,
        # Per-step reward diagnostics
        reward_breakdown     = reward_breakdown,
        # Grader fields: populated at terminal step only, None otherwise
        grader_final_score               = grader_result.get("final_score"),
        grader_conversion_score          = grader_result.get("conversion_score"),
        grader_utilization_score         = grader_result.get("utilization_score"),
        grader_bid_efficiency            = grader_result.get("bid_efficiency"),
        grader_total_conversions         = grader_result.get("total_conversions"),
        grader_total_spend               = grader_result.get("total_spend"),
        grader_budget_remaining          = grader_result.get("budget_remaining"),
        grader_max_possible_conversions  = grader_result.get("max_possible_conversions"),
    )
