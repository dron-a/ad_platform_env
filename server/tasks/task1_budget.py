# server/budget_environment.py

import numpy as np
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState
from ..grader import *


# ---------------- RESET ----------------
def reset(state: AdPlatformState, realism_mode: str | None = None, profile: dict | None = None):
    """
    Task 1 reset.
    Persists grader scores, resets all accumulators.
    """

    s = state

    # --- Persist grader scores from completed episode ---
    if s.step_count > 0:
        grader_result            = compute_score(s)
        s.prev_conversion_score  = grader_result["conversion_score"]
        s.prev_utilization_score = grader_result["utilization_score"]
        s.prev_bid_efficiency    = 0.0
        s.prev_final_score       = grader_result["final_score"]
        s.prev_episode_graded    = True

    # --- Apply historical campaign profile first (overrides defaults with real data) ---
    s.apply_profile(profile)
    if realism_mode is not None:
        s.realism_mode = realism_mode

    # --- Wipe episode state ---
    s.step_count        = 0
    s.remaining_budget  = s.total_budget
    s.total_conversions = 0.0
    s.total_spend       = 0.0
    s.spend_history.clear()
    s.obs_history.clear()
    s.reward_buffer.clear()

    # --- Reset accumulators ---
    s.cumulative_pacing_score = 0.0   # smoothness accumulator for task 1
    s.cumulative_bid_quality  = 0.0   # not used in task 1

    # --- Set conversion rates per realism mode ---
    np.random.seed(s.seed)
    if s.realism_mode == "fixed":
        s.conversion_rates = s.base_conversion_rates.copy()
    elif s.realism_mode == "realistic":
        s.conversion_rates = [
            r * (1 + 0.01 * ((i + s.seed) % 3))
            for i, r in enumerate(s.base_conversion_rates)
        ]
    else:
        raise ValueError(f"Unknown realism_mode: {s.realism_mode}")

    return AdPlatformObservation(
        step                   = s.step_count,
        total_budget           = s.total_budget,
        remaining_budget       = s.remaining_budget,
        campaign_performance   = s.conversion_rates,
        obs_history            = [],
        prev_conversion_score  = s.prev_conversion_score,
        prev_utilization_score = s.prev_utilization_score,
        prev_bid_efficiency    = s.prev_bid_efficiency,
        prev_final_score       = s.prev_final_score,
        prev_episode_graded    = s.prev_episode_graded,
    )


# ---------------- STEP ----------------
def step(state: AdPlatformState, action: AdPlatformAction):
    """
    Task 1 step function with normalized reward.
    Carryover penalty added.
    """

    s = state
    s.set_conversion_rates()

    assert len(action.allocations) == len(s.conversion_rates), "Allocation length mismatch"

    # Determine pacing limit
    if s.step_count == s.max_steps - 1:
        pacing_limit = s.remaining_budget
    else:
        pacing_limit = s.max_fraction_per_step * s.remaining_budget

    # --- Illegal allocation penalty ---
    illegal_penalty = 0.0
    for a in action.allocations:
        if a < 0:
            illegal_penalty += 0.5
        elif a > pacing_limit:
            illegal_penalty += 0.2

    # --- Clamp allocations ---
    allocations = [max(0.0, min(a, pacing_limit)) for a in action.allocations]

    # --- Budget spend ---
    budget_spent        = min(sum(allocations), s.remaining_budget)
    total_available     = s.remaining_budget
    s.remaining_budget -= budget_spent
    s.total_spend      += budget_spent
    s.spend_history.append(budget_spent)
    spend_ratio = budget_spent / (total_available + 1e-8) if total_available > 0 else 0.0

    # --- Compute conversions (seasonal multiplier applied if profile supplies one) ---
    seasonal    = s.get_seasonal_multiplier()
    conversions = sum(a * c * seasonal for a, c in zip(allocations, s.conversion_rates))
    s.total_conversions += conversions

    # --- Record step ---
    s.record_step(
        spend=budget_spent, conversions=conversions,
        allocations=allocations, bids=[]
    )

    # --- Delayed reward ---
    s.reward_buffer.append(conversions)
    if len(s.reward_buffer) == s.reward_buffer.maxlen:
        delayed_reward = s.reward_buffer.popleft()
    else:
        delayed_reward = 0.0

    # --- Spend penalty ---
    fraction_spent = budget_spent / (s.total_budget + 1e-8)
    spend_penalty = s.penalty_alpha * (fraction_spent ** s.penalty_beta)

    # --- Carryover penalty ---
    if s.step_count < s.max_steps - 1:
        time_factor       = s.step_count / s.max_steps
        carryover_penalty = 0.2 * (spend_ratio ** 2) * (1 - time_factor)
    else:
        carryover_penalty = 0.0

    # --- Accumulate smoothness into cumulative_pacing_score  ---
    smoothness = compute_smoothness(budget_spent, s.total_spend, s.step_count)
    s.cumulative_pacing_score += smoothness

    # --- Terminal check ---
    s.step_count += 1
    done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

    # --- Terminal: grader for observation only ---
    grader_result = {}
    if done:
        grader_result = compute_score(s)

    # --- Normalized step reward ---
    max_possible = s.total_budget * max(s.conversion_rates, default=1.0)
    reward_breakdown = compute_task1_step_reward(
        delayed_reward    = delayed_reward,
        spend_penalty     = spend_penalty,
        carryover_penalty = carryover_penalty,
        illegal_penalty   = illegal_penalty,
        spend             = budget_spent,
        total_spend       = s.total_spend,
        total_budget      = s.total_budget,
        step_count        = s.step_count,
        is_terminal       = done,
        cumulative_smooth = s.cumulative_pacing_score,
        cumulative_conv   = s.total_conversions,
        max_possible_conv = max_possible,
    )
    reward = reward_breakdown["step_reward"]
    s.last_step_reward = reward

    return AdPlatformObservation(
        step                 = s.step_count,
        total_budget         = s.total_budget,
        remaining_budget     = s.remaining_budget,
        campaign_performance = s.conversion_rates,
        obs_history          = list(s.obs_history),
        reward               = reward,
        done                 = done,
        reward_breakdown     = reward_breakdown,
        grader_final_score               = grader_result.get("final_score"),
        grader_conversion_score          = grader_result.get("conversion_score"),
        grader_utilization_score         = grader_result.get("utilization_score"),
        grader_bid_efficiency            = None,   # not applicable for task 1
        grader_total_conversions         = grader_result.get("total_conversions"),
        grader_total_spend               = grader_result.get("total_spend"),
        grader_budget_remaining          = grader_result.get("budget_remaining"),
        grader_max_possible_conversions  = grader_result.get("max_possible_conversions"),
    )