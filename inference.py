"""
inference.py
============
Inference script for AdPlatform RL environment.
Configured for auction task by default.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL      : LLM API endpoint
    MODEL_NAME        : model identifier
    HF_TOKEN          : Hugging Face / API key
    IMAGE_NAME        : docker image name (if using from_docker_image)
    TASK_NAME         : budget | auction | dynamic_campaign (default: auction)
    YAML_PATH         : path to Ads export YAML (optional)

STDOUT FORMAT:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Score: mean normalized reward per step → sum(rewards) / MAX_STEPS
       Each step reward is already in [0, 1] from reward system.
       Final score is therefore in [0, 1].
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.environment import AdPlatformEnvironment
from models import AdPlatformAction, CampaignProfile
from client import AdPlatformClient

# ---------------- CONFIG ----------------
IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("TASK",    "auction")
YAML_PATH    = os.getenv("YAML_PATH",    None)
BENCHMARK    = os.getenv("BENCHMARK",    "ad_platform_env")

MAX_STEPS               = 30
TEMPERATURE             = 0.7
MAX_TOKENS              = 256
SUCCESS_SCORE_THRESHOLD = 0.5   # normalized score in [0, 1]

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous bidding agent controlling an ad platform environment.
    Your goal is to maximize conversions by allocating budget and setting bids
    across 3 campaigns over 30 steps.

    Respond with a valid JSON object only — no commentary, no markdown, no extra text.

    Action format:
      {"allocations": [<float>, <float>, <float>], "bids": [<float>, <float>, <float>]}

    Guidelines:
    - CRITICAL: You have exactly 30 steps. Pace your budget evenly — spend 
                roughly 1/30 of remaining budget per step. Never spend more than 10% 
                of total budget in a single step. Spending all budget early leaves 
                nothing for the remaining steps and destroys your score.

    - allocations : budget to spend per campaign this step (non-negative floats)
    - bids        : bid price per campaign — bid above competitor bids to win auctions

    - KEY STRATEGIES
    - Concentrate spend and high bids on campaigns with higher conversion rates
    - If you won fewer auctions last step, raise bids on high-converting campaigns
    - If your previous episode score was low, adjust your overall strategy this episode
    - Pace your budget — avoid dumping it all early, save some for later steps
    """
).strip()


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
    competitor_bids: Optional[list] = None,
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
    if competitor_bids:
        print(f" competitor_bids={competitor_bids}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------- PROMPT ----------------
def build_user_prompt(
    step:         int,
    obs:          dict,
    last_reward:  float,
) -> str:
    """
    Build the user prompt for the model.
    Uses obs_history from observation for structured step context.
    Includes prev episode grader scores for episodic policy conditioning.
    """
    # --- Current state ---
    total_budget     = obs.get("total_budget", 1000.0)
    remaining_budget = obs.get("remaining_budget", 0.0)
    budget_spent     = total_budget - remaining_budget
    budget_pct_used  = (budget_spent / (total_budget + 1e-8)) * 100
    campaign_perf     = obs.get("campaign_performance", [])
    competitor_bids   = obs.get("competitor_bids", [])

    # --- Obs history window (last history_window steps from environment) ---
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

    # --- Previous episode grader scores ---
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

    return textwrap.dedent(
        f"""
        Step: {step} / 30
        Total budget       : {total_budget:.2f}
        Remaining budget   : {remaining_budget:.2f}
        Budget spent       : {budget_spent:.2f} ({budget_pct_used:.1f}% used)
        Campaign conv rates: {campaign_perf}
        Competitor bids    : {competitor_bids}
        Last step reward   : {last_reward:.4f}

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
) -> dict:
    prompt = build_user_prompt(step, obs, last_reward)
    try:
        # print(f"[DEBUG] Prompt:\n{prompt}", flush=True)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # print(f"[DEBUG] Model raw response: {text}", flush=True)
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


# ---------------- MAIN LOOP ----------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Instantiate environment
    if IMAGE_NAME:
        env = await AdPlatformClient.from_docker_image(
            IMAGE_NAME,
            env_vars={"TASK": TASK_NAME}
        )
    else:
        env = AdPlatformEnvironment(task=TASK_NAME, yaml_path=YAML_PATH)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # --- Reset ---
        result = env.reset()
        if asyncio.iscoroutine(result):
            result = await result
        last_reward = 0.0
        # Extract observation from StepResult wrapper
        obs_obj = getattr(result, "observation", result)
        obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else {}
        # obs = result.model_dump() if hasattr(result, "model_dump") else {}


        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10
        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            # --- Get model action ---
            action_dict = get_model_action(client, step, obs, last_reward)

            # --- Early termination if model fails too many times ---
            if not action_dict:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"[DEBUG] {consecutive_failures} consecutive model failures — terminating early", flush=True)
                    break
            else:
                consecutive_failures = 0

            # --- Defaults for missing fields ---
            if "allocations" not in action_dict:
                action_dict["allocations"] = [0.0, 0.0, 0.0]
            if "bids" not in action_dict:
                # Must be length 3 for auction/dynamic_campaign
                # Zero bids lose all auctions but prevent assertion failure
                action_dict["bids"] = [0.0, 0.0, 0.0]

            # --- Validate and create action ---
            try:
                action_obj = AdPlatformAction(**action_dict)
            except Exception as exc:
                print(f"[DEBUG] Invalid action, using zeros: {exc}", flush=True)
                action_obj = AdPlatformAction(
                    allocations=[0.0, 0.0, 0.0],
                    bids=[0.0, 0.0, 0.0]
                )

            # --- Step ---
            result = env.step(action_obj)
            if asyncio.iscoroutine(result):
                result = await result
            reward = getattr(result, "reward", 0.0) or 0.0
            done   = getattr(result, "done",   False)
            # Extract observation from StepResult wrapper
            obs_obj = getattr(result, "observation", result)
            obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else {}
            # obs    = result.model_dump() if hasattr(result, "model_dump") else {}

            # --- Extract reward breakdown for diagnostics ---
            breakdown = obs.get("reward_breakdown")

            steps_taken  = step
            last_reward  = reward
            rewards.append(reward)
            competitor_bids=getattr(obs_obj, "competitor_bids", None)

            log_step(
                step      = step,
                action    = str(action_dict),
                reward    = reward,
                done      = done,
                error     = None,
                breakdown = breakdown,
                competitor_bids = competitor_bids
            )

            if done:
                break

        # --- Score: mean normalized reward per step → [0, 1] ---
        score   = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score   = float(max(0.0, min(1.0, score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if hasattr(env, "close"):
                close_result = env.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
