"""
inference.py
============
Inference script for AdPlatform RL environment.
Runs all three tasks sequentially with continuous stdout output.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL      : LLM API endpoint
    MODEL_NAME        : model identifier
    HF_TOKEN          : Hugging Face / API key
    IMAGE_NAME        : docker image name (if using from_docker_image)
    YAML_PATH         : path to Ads export YAML (optional)

STDOUT FORMAT (per task, continuous):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Score: mean normalized reward per step -> sum(rewards) / MAX_STEPS
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from server.environment import AdPlatformEnvironment
from models import AdPlatformAction
from client import AdPlatformClient

# ---------------- CONFIG ----------------
IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
YAML_PATH    = os.getenv("YAML_PATH",    None)
BENCHMARK    = os.getenv("BENCHMARK",    "ad_platform_env")

# All three tasks run sequentially
ALL_TASKS = ["budget", "auction", "dynamic_campaign"]

MAX_STEPS               = 30
TEMPERATURE             = 0.7
MAX_TOKENS              = 256
SUCCESS_SCORE_THRESHOLD = 0.5


# ---------------- SYSTEM PROMPTS ----------------
SYSTEM_PROMPTS = {
    "budget": textwrap.dedent("""
        You are an autonomous budget allocation agent controlling an ad platform.
        Your goal is to maximize conversions by allocating budget across 3 campaigns over 30 steps.
        No bidding required.

        Respond with a valid JSON object only — no commentary, no markdown, no extra text.

        Action format:
          {"allocations": [<float>, <float>, <float>], "bids": [0.0, 0.0, 0.0]}

        Guidelines:
        - CRITICAL: Pace your budget evenly — spend roughly 1/30 of remaining budget per step.
        - Concentrate spend on campaigns with higher conversion rates.
        - Never spend more than 10% of total budget in a single step.
        - Use the full budget across all 30 steps for best score.
    """).strip(),

    "auction": textwrap.dedent("""
        You are an autonomous bidding agent controlling an ad platform.
        Your goal is to maximize conversions by allocating budget and setting bids
        across 3 campaigns over 30 steps.

        Respond with a valid JSON object only — no commentary, no markdown, no extra text.

        Action format:
          {"allocations": [<float>, <float>, <float>], "bids": [<float>, <float>, <float>]}

        Guidelines:
        - CRITICAL: Pace your budget evenly — spend roughly 1/30 of remaining budget per step.
        - Never spend more than 10% of total budget in a single step.
        - Bid above competitor bids to win auctions — check competitor bids each step.
        - Concentrate spend and high bids on campaigns with higher conversion rates.
        - If your previous episode score was low, adjust your overall strategy.
    """).strip(),

    "dynamic_campaign": textwrap.dedent("""
        You are an autonomous bidding agent controlling an ad platform with dynamic campaigns.
        Your goal is to maximize conversions across 3 campaigns over 30 steps.
        Campaigns have seasonality, market events, and adaptive competitors.

        Respond with a valid JSON object only — no commentary, no markdown, no extra text.

        Action format:
          {"allocations": [<float>, <float>, <float>], "bids": [<float>, <float>, <float>]}

        Guidelines:
        - CRITICAL: Pace your budget evenly — spend roughly 1/30 of remaining budget per step.
        - Never spend more than 10% of total budget in a single step.
        - Watch campaign conversion rates closely — they change each step due to market events.
        - When a campaign conversion rate spikes above normal, increase spend and bids on it.
        - Bid above competitor bids to win auctions — competitors adapt to your bids.
        - Concentrate spend on the highest converting campaign each step.
    """).strip(),
}


# ---------------- LOGGING ----------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:      int,
    action:    str,
    reward:    float,
    done:      bool,
    error:     Optional[str],
    breakdown: Optional[dict] = None,
) -> None:
    breakdown_str = ""
    if breakdown:
        breakdown_str = (
            f" conv={breakdown.get('conv_component', 0):.3f}"
            f" bid={breakdown.get('bid_quality_component', 0):.3f}"
            f" pacing={breakdown.get('pacing_component', 0):.3f}"
            f" gate={breakdown.get('illegal_gate', 0):.3f}"
        )
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}{breakdown_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------- PROMPT ----------------
def build_user_prompt(
    step:        int,
    obs:         dict,
    last_reward: float,
    task:        str,
) -> str:
    total_budget     = obs.get("total_budget", 1000.0)
    remaining_budget = obs.get("remaining_budget", 0.0)
    budget_spent     = total_budget - remaining_budget
    budget_pct_used  = (budget_spent / (total_budget + 1e-8)) * 100
    suggested_spend  = remaining_budget / max(MAX_STEPS - step + 1, 1)
    campaign_perf    = obs.get("campaign_performance", [])
    competitor_bids  = obs.get("competitor_bids", [])

    obs_history = obs.get("obs_history", [])
    if obs_history:
        history_lines = []
        for h in obs_history[-5:]:
            history_lines.append(
                f"  step={h.get('step')} "
                f"spend={h.get('spend')} "
                f"conversions={h.get('conversions')} "
                f"allocations={h.get('allocations')} "
                f"bids={h.get('bids')} "
                f"competitor_bids={h.get('competitor_bids')}"
            )
        history_block = "\n".join(history_lines)
    else:
        history_block = "None"

    prev_graded = obs.get("prev_episode_graded", False)
    if prev_graded:
        prev_block = textwrap.dedent(
            f"""
            Previous episode performance:
              Overall score    : {obs.get('prev_final_score', 0.0):.3f}
              Conversion score : {obs.get('prev_conversion_score', 0.0):.3f}
              Utilization score: {obs.get('prev_utilization_score', 0.0):.3f}
              Bid efficiency   : {obs.get('prev_bid_efficiency', 0.0):.3f}
            """
        ).strip()
    else:
        prev_block = "Previous episode: none (first episode)"

    # Competitor bids only relevant for bidding tasks
    competitor_line = ""
    if task in ("auction", "dynamic_campaign") and competitor_bids:
        competitor_line = f"Competitor bids    : {competitor_bids}\n        "

    return textwrap.dedent(
        f"""
        Step: {step} / {MAX_STEPS}
        Total budget       : {total_budget:.2f}
        Remaining budget   : {remaining_budget:.2f}
        Budget spent       : {budget_spent:.2f} ({budget_pct_used:.1f}% used)
        Suggested spend    : {suggested_spend:.2f} (target this step)
        Campaign conv rates: {campaign_perf}
        {competitor_line}Last step reward   : {last_reward:.4f}

        Recent step history:
        {history_block}

        {prev_block}

        Provide the next action as JSON.
        """
    ).strip()


# ---------------- MODEL CALL ----------------
def get_model_action(
    client:      OpenAI,
    step:        int,
    obs:         dict,
    last_reward: float,
    task:        str,
) -> dict:
    prompt = build_user_prompt(step, obs, last_reward, task)
    system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["auction"])
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        try:
            action = json.loads(text)
            if not isinstance(action, dict):
                action = {}
        except Exception:
            action = {}
        return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {}


# ---------------- EPISODE RUNNER ----------------
async def run_episode(
    client: OpenAI,
    env:    any,
    task:   str,
) -> Tuple[bool, int, float, List[float]]:
    """
    Run one complete episode for a given task.
    Emits [START], [STEP] x MAX_STEPS, [END] to stdout.
    Returns (success, steps_taken, score, rewards).
    """
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # --- Reset ---
        if IMAGE_NAME:
            result = await env.reset()
        else:
            result = env.reset(task=task)
            if asyncio.iscoroutine(result):
                result = await result

        last_reward = 0.0
        obs_obj = getattr(result, "observation", result)
        obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else {}

        consecutive_failures    = 0
        MAX_CONSECUTIVE_FAILURES = 10

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            # --- Get model action ---
            action_dict = get_model_action(client, step, obs, last_reward, task)

            if not action_dict:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"[DEBUG] {consecutive_failures} consecutive failures — terminating early",
                        flush=True,
                    )
                    break
            else:
                consecutive_failures = 0

            # --- Defaults ---
            if "allocations" not in action_dict:
                action_dict["allocations"] = [0.0, 0.0, 0.0]
            if "bids" not in action_dict:
                action_dict["bids"] = [0.0, 0.0, 0.0]

            # --- Validate action ---
            try:
                action_obj = AdPlatformAction(**action_dict)
            except Exception as exc:
                print(f"[DEBUG] Invalid action, using zeros: {exc}", flush=True)
                action_obj = AdPlatformAction(
                    allocations=[0.0, 0.0, 0.0],
                    bids=[0.0, 0.0, 0.0],
                )

            # --- Step ---
            if IMAGE_NAME:
                result = await env.step(action_obj)
            else:
                result = env.step(action_obj)
                if asyncio.iscoroutine(result):
                    result = await result

            reward  = getattr(result, "reward", 0.0) or 0.0
            done    = getattr(result, "done",   False)
            obs_obj = getattr(result, "observation", result)
            obs     = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else {}
            breakdown = obs.get("reward_breakdown")

            steps_taken = step
            last_reward = reward
            rewards.append(reward)

            log_step(
                step      = step,
                action    = str(action_dict),
                reward    = reward,
                done      = done,
                error     = None,
                breakdown = breakdown,
            )

            if done:
                break

        # --- Score ---
        score   = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score   = float(max(0.0, min(1.0, score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


# ---------------- MAIN ----------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in ALL_TASKS:
        if IMAGE_NAME:
            # Fresh container per task with correct TASK env var
            # Container lifecycle is silent — no output between tasks
            env = await AdPlatformClient.from_docker_image(
                IMAGE_NAME,
                env_vars={"TASK": task},
            )
            try:
                await run_episode(client, env, task)
            finally:
                try:
                    await env.close()
                except Exception:
                    pass   # silent cleanup
        else:
            # Local — fresh env per task
            env = AdPlatformEnvironment(task=task, yaml_path=YAML_PATH)
            try:
                await run_episode(client, env, task)
            except Exception as e:
                print(f"[DEBUG] Episode error for task={task}: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
