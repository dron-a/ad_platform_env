from models import AdPlatformState
import numpy as np


def compute_score(state) -> dict:
    """
    Grader for task 1: budget allocation.
    Returns breakdown dict with final_score in [0, 1].

    Dimensions:
      conversion_score  (0.70): total_conversions / theoretical max
      utilization_score (0.20): how fully the budget was used
      smoothness_score  (0.10): consistency of spend across steps
    """
    
    # --- Conversion score ---
    max_possible     = state.total_budget * max(state.conversion_rates, default=1.0)
    conversion_score = float(max(0.0, min(1.0,
        state.total_conversions / (max_possible + 1e-8)
    )))
    
    # --- Budget utilization ---
    utilization       = state.total_spend / (state.total_budget + 1e-8)
    utilization_score = float(max(0.0, 1.0 - abs(1.0 - utilization)))

    # --- Smooth pacing ---
    if len(state.spend_history) > 1:
        spend_std = np.std(state.spend_history)
        spend_mean = np.mean(state.spend_history) + 1e-8
        smoothness_score = 1 - (spend_std / spend_mean)
        smoothness_score = float(max(0.0, smoothness_score))
    else:
        smoothness_score = 0.0

    # --- Final weighted score ---
    final_score = (
        0.70 * conversion_score +
        0.20 * utilization_score +
        0.10 * smoothness_score
    )
    final_score = float(max(0.0, min(1.0, final_score)))

    return {
        "final_score":               final_score,
        "conversion_score":          round(conversion_score,  4),
        "utilization_score":         round(utilization_score, 4),
        "smoothness_score":          round(smoothness_score,  4),
        "total_conversions":         state.total_conversions,
        "total_spend":               state.total_spend,
        "budget_remaining":          state.remaining_budget,
        "max_possible_conversions":  round(max_possible, 4),
    }


def compute_auction_score(state: AdPlatformState) -> dict:
    """
    Episode-level grader. Returns scores as OBSERVATION FIELDS ONLY.
    Not used in reward calculation.
 
    Scores:
      conversion_score  (0.60): total_conversions / theoretical max
      utilization_score (0.25): how fully the budget was used
      bid_efficiency    (0.15): conversions per unit spend, normalised
    """
# --- Conversion score ---
    # Theoretical max: spend entire budget on the best-converting campaign
    # and win every auction (win_prob → 1 when bid >> competitor)
    max_possible = state.total_budget * max(state.conversion_rates, default=1.0)
    conversion_score = state.total_conversions / (max_possible + 1e-8)
    conversion_score = float(max(0.0, min(1.0, conversion_score)))

    # --- Budget utilisation ---
    utilization = state.total_spend / (state.total_budget + 1e-8)
    utilization_score = 1.0 - abs(1.0 - utilization)
    utilization_score = float(max(0.0, utilization_score))

    # --- Bid efficiency: conversions per unit spend ---
    # Normalised against the best conversion rate so scale is comparable
    best_cr = max(state.conversion_rates, default=1.0)
    if state.total_spend > 0:
        actual_cps = state.total_conversions / state.total_spend          # conversions per $
        ideal_cps = best_cr                                                 # upper bound
        bid_efficiency = actual_cps / (ideal_cps + 1e-8)
        bid_efficiency = float(max(0.0, min(1.0, bid_efficiency)))
    else:
        bid_efficiency = 0.0

    # --- Weighted final score ---
    final_score = (
        0.60 * conversion_score +
        0.25 * utilization_score +
        0.15 * bid_efficiency
    )
    final_score = float(max(0.0, min(1.0, final_score)))

    return {
        "final_score": final_score,
        "conversion_score": round(conversion_score, 4),
        "utilization_score": round(utilization_score, 4),
        "bid_efficiency": round(bid_efficiency, 4),
        # Raw stats for debugging reward alignment
        "total_conversions": state.total_conversions,
        "total_spend": state.total_spend,
        "budget_remaining": state.remaining_budget,
        "max_possible_conversions": round(max_possible, 4),
    }


def compute_dynamics_campaign_score(state: AdPlatformState) -> dict:
    """
    Independent evaluation of dynamic_campaign task performance.
    Returns a breakdown dict and a final score in [0, 1].

    Scoring dimensions:
      - conversion_score  (0.50): total_conversions / theoretical maximum
      - utilization_score (0.25): how fully the budget was used
      - bid_efficiency    (0.15): conversions per unit spend (normalised)
      - adaptability      (0.10): reward for exploiting market events
                                  (score is higher when conversions spike at known event steps)
    """

    # --- Conversion score ---
    # Theoretical max uses the highest base conversion rate × full budget
    max_possible = state.total_budget * max(state.base_conversion_rates, default=1.0)
    conversion_score = state.total_conversions / (max_possible + 1e-8)
    conversion_score = max(0.0, min(1.0, conversion_score))

    # --- Budget utilisation ---
    utilization = state.total_spend / (state.total_budget + 1e-8)
    utilization_score = 1.0 - abs(1.0 - utilization)
    utilization_score = max(0.0, utilization_score)

    # --- Bid efficiency ---
    best_cr = max(state.base_conversion_rates, default=1.0)
    if state.total_spend > 0:
        actual_cps = state.total_conversions / state.total_spend
        bid_efficiency = actual_cps / (best_cr + 1e-8)
        bid_efficiency = max(0.0, min(1.0, bid_efficiency))
    else:
        bid_efficiency = 0.0

    # --- Adaptability: did the agent spend more during market event steps? ---
    # Market events occur at steps 10 and 20 (multipliers > 1 for some campaigns).
    # Proxy: check if spend_history shows relatively higher spend at those steps.
    event_steps = set(state.market_events.keys())  # {10, 20}
    if len(state.spend_history) > 0 and len(event_steps) > 0:
        mean_spend = np.mean(state.spend_history) + 1e-8
        event_spends = [
            state.spend_history[s] for s in event_steps
            if s < len(state.spend_history)
        ]
        if event_spends:
            # Score > 0.5 if event-step spend exceeds average
            adaptability = min(1.0, np.mean(event_spends) / (mean_spend * 1.2))
        else:
            adaptability = 0.5  # no data yet, neutral
    else:
        adaptability = 0.0

    # --- Weighted final score ---
    final_score = (
        0.50 * conversion_score +
        0.25 * utilization_score +
        0.15 * bid_efficiency +
        0.10 * adaptability
    )
    final_score = float(max(0.0, min(1.0, final_score)))

    return {
        "final_score": final_score,
        "conversion_score": round(conversion_score, 4),
        "utilization_score": round(utilization_score, 4),
        "bid_efficiency": round(bid_efficiency, 4),
        "adaptability_score": round(adaptability, 4),
        # Raw stats
        "total_conversions": state.total_conversions,
        "total_spend": state.total_spend,
        "budget_remaining": state.remaining_budget,
        "max_possible_conversions": round(max_possible, 4),
    }
