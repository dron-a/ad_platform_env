# server/dynamic_campaign_environment.py

import numpy as np
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState
from ..grader import *


# ---------------- RESET ----------------
def reset(state: AdPlatformState, profile: dict | None = None):
    """Task 3 reset — persists grader scores, resets all task 3 accumulators."""

    s = state

    # --- Persist grader scores from completed episode ---
    if s.step_count > 0:
        grader_result            = compute_dynamics_campaign_score(s)
        s.prev_conversion_score  = grader_result["conversion_score"]
        s.prev_utilization_score = grader_result["utilization_score"]
        s.prev_bid_efficiency    = grader_result["bid_efficiency"]
        s.prev_final_score       = grader_result["final_score"]
        s.prev_episode_graded    = True

    # --- Apply historical campaign profile (overrides defaults with real data) ---
    s.apply_profile(profile)
    s.reward_bounds = compute_reward_bounds(s)
    s.step_count               = 0
    s.remaining_budget         = s.total_budget
    s.total_conversions        = 0.0
    s.total_spend              = 0.0
    s.spend_history.clear()
    s.obs_history.clear()
    s.reward_buffer.clear()
    s.conversion_rates         = s.base_conversion_rates.copy()
    s.prev_agent_bids          = [0.0] * len(s.conversion_rates)
    s.competitor_bids          = s.base_competitor_bids.copy()

    # --- Default market events ---
    if not s.market_events:
        s.market_events = {
            10: [1.2, 1.0, 0.8],
            20: [0.9, 1.3, 1.0],
        }

    # --- Reset task 3 accumulators ---
    s.cumulative_bid_quality          = 0.0
    s.cumulative_pacing_score         = 0.0
    s.cumulative_adaptability_score   = 0.0
    s.episode_mean_opportunity        = 0.0

    return AdPlatformObservation(
        step                 = s.step_count,
        total_budget         = s.total_budget,
        remaining_budget     = s.remaining_budget,
        campaign_performance = s.conversion_rates,
        competitor_bids      = s.competitor_bids,
        obs_history          = [],
        prev_conversion_score  = s.prev_conversion_score,
        prev_utilization_score = s.prev_utilization_score,
        prev_bid_efficiency    = s.prev_bid_efficiency,
        prev_final_score       = s.prev_final_score,
        prev_episode_graded    = s.prev_episode_graded,
    )


# ---------------- STEP ----------------
def step(state: AdPlatformState, action: AdPlatformAction) -> AdPlatformObservation:
    s = state

    allocations = action.allocations
    bids = action.bids

    assert len(bids) == len(s.base_conversion_rates), "Bid length mismatch"
    assert len(allocations) == len(s.base_conversion_rates), "Allocation length mismatch"

    # ----------------------------
    # Update conversion rates (profile seasonal + market events + seeded noise)
    # ----------------------------
    seasonal = s.get_seasonal_multiplier()
    updated_rates = []
    for i, base in enumerate(s.base_conversion_rates):
        event_multiplier = s.market_events.get(
            s.step_count, [1.0] * len(s.base_conversion_rates)
        )[i]
        rng = np.random.default_rng(s.seed + s.step_count + i)
        noise = rng.uniform(-0.02, 0.02)
        updated_rates.append(base * seasonal * event_multiplier * (1 + noise))

    s.conversion_rates = updated_rates

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
    spend = sum(allocations)
    spend = min(spend, s.remaining_budget)
    total_available = s.remaining_budget
    s.remaining_budget -= spend
    s.total_spend += spend

    spend_ratio = spend / (total_available + 1e-8) if total_available > 0 else 0.0
    s.spend_history.append(spend)

    # ----------------------------
    # Conversions (win probability via sigmoid)
    # ----------------------------
    conversions = 0.0
    for a, bid, cb, cr in zip(allocations, bids, s.competitor_bids, s.conversion_rates):
        win_prob     = 1.0 / (1.0 + np.exp(cb - bid))
        conversions += a * cr * win_prob
    s.total_conversions += conversions

    # Record step into rolling history
    s.record_step(spend=spend, conversions=conversions, allocations=allocations, bids=bids)

    # ----------------------------
    # Delayed reward
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
    # Carryover penalty
    # ----------------------------
    carryover_penalty = 0.0
    if s.step_count < s.max_steps - 1:
        time_factor = s.step_count / s.max_steps
        carryover_penalty = 0.2 * (spend_ratio ** 2) * (1 - time_factor)

    # --- Step opportunity and episode mean ---
    step_opp = compute_step_opportunity(s.conversion_rates, s.competitor_bids)
    steps_so_far = s.step_count + 1
    s.episode_mean_opportunity = (
        (s.episode_mean_opportunity * s.step_count + step_opp) / steps_so_far
    )

    # --- Accumulate signals ---
    bid_quality = compute_bid_quality(
        bids, s.competitor_bids, s.conversion_rates, allocations
    )
    pacing_signal = compute_opportunity_pacing(
        spend, total_available, s.step_count, s.max_steps,
        step_opp, s.episode_mean_opportunity,
    )
    adaptability_signal = compute_adaptability(
        s.conversion_rates, s.base_conversion_rates,
        allocations, bids, s.competitor_bids,
    )

    s.cumulative_bid_quality          += bid_quality
    s.cumulative_pacing_score         += pacing_signal
    s.cumulative_adaptability_score   += adaptability_signal

    # --- Terminal check ---
    s.prev_agent_bids = bids.copy()
    s.step_count += 1
    done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

    # --- Terminal: grader for observation only ---
    grader_result = {}
    if done:
        grader_result = compute_dynamics_campaign_score(s)

    # --- Normalized step reward ---
    max_possible = s.total_budget * max(s.base_conversion_rates, default=1.0)
    reward_breakdown = compute_task3_step_reward(
        delayed_reward           = delayed_reward,
        bids                     = bids,
        competitor_bids          = s.competitor_bids,
        conversion_rates         = s.conversion_rates,
        base_conversion_rates    = s.base_conversion_rates,
        allocations              = allocations,
        spend                    = spend,
        remaining_budget         = total_available,
        spend_penalty            = spend_penalty,
        carryover_penalty        = carryover_penalty,
        illegal_penalty          = illegal_penalty,
        step_count               = s.step_count,
        max_steps                = s.max_steps,
        step_opportunity         = step_opp,
        episode_mean_opportunity = s.episode_mean_opportunity,
        is_terminal              = done,
        cumulative_pacing        = s.cumulative_pacing_score,
        cumulative_bid           = s.cumulative_bid_quality,
        cumulative_conv          = s.total_conversions,
        cumulative_adaptability  = s.cumulative_adaptability_score,
        max_possible_conv        = max_possible,
        bounds  = s.reward_bounds,
    )
    reward = reward_breakdown["step_reward"]
    s.last_step_reward = reward

    return AdPlatformObservation(
        step                 = s.step_count,
        total_budget         = s.total_budget,
        remaining_budget     = s.remaining_budget,
        campaign_performance = s.conversion_rates,
        competitor_bids      = s.competitor_bids,
        obs_history          = list(s.obs_history),
        reward               = reward,
        done                 = done,
        reward_breakdown     = reward_breakdown,
        grader_final_score               = grader_result.get("final_score"),
        grader_conversion_score          = grader_result.get("conversion_score"),
        grader_utilization_score         = grader_result.get("utilization_score"),
        grader_bid_efficiency            = grader_result.get("bid_efficiency"),
        grader_total_conversions         = grader_result.get("total_conversions"),
        grader_total_spend               = grader_result.get("total_spend"),
        grader_budget_remaining          = grader_result.get("budget_remaining"),
        grader_max_possible_conversions  = grader_result.get("max_possible_conversions"),
    )