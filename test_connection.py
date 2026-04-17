"""
test_connection.py
==================
End-to-end connection test for AdPlatform RL environment.

Tests:
  1. Market provider — data source, profile fields, benchmark ranges
  2. Docker connection — WebSocket established, correct URL
  3. Reset — raw wire, observation fields, initial state
  4. One step — raw wire, reward range, breakdown, obs_history
  5. Task-specific checks — competitor_bids presence per task

Usage:
    # Minimal — auction task, no market data
    IMAGE_NAME=adplatform_env:latest python test_connection.py

    # Full — with market data
    IMAGE_NAME=adplatform_env:latest MARKET_VERTICAL=ecommerce python test_connection.py

    # Different task
    IMAGE_NAME=adplatform_env:latest TASK=budget python test_connection.py

Environment variables:
    IMAGE_NAME            : docker image to test (required)
    TASK                  : budget | auction | dynamic_campaign (default: auction)
    MARKET_VERTICAL       : ecommerce | saas | travel | finance (optional)
    MARKET_SAMPLING       : sequential | random (default: sequential)
    MARKET_REFRESH_PER_EPISODE: true | false (default: true)
    FRED_DATA_API_KEY     : FRED API key (optional)
    YAML_PATH             : path to user YAML (optional)
"""

import asyncio
import json
import os
import re
import types
from typing import Dict

from client import AdPlatformClient
from models import AdPlatformAction

# ---------------- CONFIG ----------------
IMAGE_NAME        = os.getenv("IMAGE_NAME")
TASK              = os.getenv("TASK",             "auction")
VERTICAL          = os.getenv("MARKET_VERTICAL",  None)
SAMPLING          = os.getenv("MARKET_SAMPLING",  "sequential")
REFRESH           = os.getenv("MARKET_REFRESH_PER_EPISODE", "true")
FRED_DATA_API_KEY = os.getenv("FRED_DATA_API_KEY", "")
YAML_PATH         = os.getenv("YAML_PATH", "")

# Tasks that should have competitor_bids populated
BIDDING_TASKS = {"auction", "dynamic_campaign"}

# Expected reward breakdown keys
EXPECTED_BREAKDOWN_KEYS = {"step_reward", "illegal_gate"}


# ---------------- HELPERS ----------------
def passed(label: str) -> None:
    print(f"  [PASS] {label}", flush=True)


def failed(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}{f' — {detail}' if detail else ''}", flush=True)


def check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        passed(label)
    else:
        failed(label, detail)
    return condition


def section(title: str) -> None:
    print(f"\n{'='*50}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*50}", flush=True)


# ---------------- RAW WIRE PATCH ----------------
async def debug_receive(self):
    """
    Intercepts raw WebSocket message before JSON parsing.
    Prints raw wire data for full transparency.
    """
    assert self._ws is not None
    raw = await asyncio.wait_for(
        self._ws.recv(), timeout=self._message_timeout
    )
    match = re.search(r'"competitor_bids"\s*:\s*(\[[^\]]*\])', raw)
    print(
        f"  [WIRE] competitor_bids = "
        f"{match.group(1) if match else 'NOT FOUND IN RAW'}",
        flush=True,
    )
    print(f"  [WIRE] first 300 chars: {raw[:300]}", flush=True)
    return json.loads(raw)


# ---------------- TEST 1: MARKET PROVIDER ----------------
def test_market_provider() -> bool:
    section("TEST 1 — Market Provider")

    if not VERTICAL:
        print("  Skipped — MARKET_VERTICAL not set", flush=True)
        print("  Environment will use default_profile.yaml or synthetic defaults", flush=True)
        return True

    try:
        from .data_build import MarketDataProvider
    except ImportError:
        from data_build import MarketDataProvider
        # return False

    try:
        provider = MarketDataProvider(
            vertical=VERTICAL,
            refresh_per_episode=REFRESH.lower() == "true",
            sampling=SAMPLING,
        )
    except Exception as e:
        failed("MarketDataProvider init", str(e))
        return False

    check(True, f"Provider created for vertical={VERTICAL}")
    check(provider.source != "none", f"Data source used: {provider.source}")
    check(provider.data_date_range != "unknown", f"Date range: {provider.data_date_range}")

    try:
        profile = provider.get_profile()
    except Exception as e:
        failed("get_profile()", str(e))
        return False

    check(True, "get_profile() succeeded")

    # Check all expected profile fields
    seasonal = profile.get("seasonal_multipliers", [])
    check(len(seasonal) == 30, f"seasonal_multipliers has 30 values", f"got {len(seasonal)}")

    cvrs = profile.get("conversion_rates", [])
    check(len(cvrs) == 3, "conversion_rates has 3 values", f"got {len(cvrs)}")
    check(all(0 < v < 1 for v in cvrs), "conversion_rates in (0, 1)", str(cvrs))

    cbids = profile.get("competitor_bids", [])
    check(len(cbids) == 3, "competitor_bids has 3 values", f"got {len(cbids)}")
    check(all(v > 0 for v in cbids), "competitor_bids all positive", str(cbids))

    bvol = profile.get("bid_volatility", [])
    check(len(bvol) == 3, "bid_volatility has 3 values", f"got {len(bvol)}")

    budget = profile.get("total_budget", 0)
    check(budget > 0, f"total_budget > 0: {budget}")

    events = profile.get("market_events", {})
    print(f"  [INFO] market_events detected: {len(events)} events at steps {list(events.keys())}", flush=True)

    return True


# ---------------- TEST 2-5: DOCKER CONNECTION ----------------
async def test_docker(env_vars: dict) -> None:
    section("TEST 2 — Docker Connection")

    print(f"  Env vars passed to container:", flush=True)
    for k, v in env_vars.items():
        # Mask sensitive values
        display_v = "***" if k in ("FRED_DATA_API_KEY",) and v else v or "(empty)"
        print(f"    {k}={display_v}", flush=True)

    env = await AdPlatformClient.from_docker_image(IMAGE_NAME, env_vars=env_vars)
    check(env._ws is not None, "WebSocket connected")
    check(env._ws_url.startswith("ws://"), f"WebSocket URL: {env._ws_url}")

    # Patch _receive for raw wire inspection
    env._receive = types.MethodType(debug_receive, env)

    try:
        await test_reset(env)
        await test_step(env)
    finally:
        try:
            await env.close()
        except Exception:
            pass


# ---------------- TEST 3: RESET ----------------
async def test_reset(env: AdPlatformClient) -> None:
    section("TEST 3 — Reset")

    try:
        result = await env.reset()
    except Exception as e:
        failed("reset() call", str(e))
        return

    check(result is not None, "reset() returned result")

    obs = getattr(result, "observation", result)
    check(obs is not None, "observation present")

    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else {}

    # Step and budget checks
    step = obs_dict.get("step", -1)
    check(step == 0, f"step=0 at reset", f"got {step}")

    total_budget     = obs_dict.get("total_budget", 0)
    remaining_budget = obs_dict.get("remaining_budget", 0)
    check(total_budget > 0, f"total_budget > 0: {total_budget}")
    check(
        abs(total_budget - remaining_budget) < 0.01,
        "remaining_budget == total_budget at reset",
        f"total={total_budget} remaining={remaining_budget}",
    )

    # Campaign performance
    camp_perf = obs_dict.get("campaign_performance", [])
    check(len(camp_perf) == 3, "campaign_performance has 3 values", f"got {len(camp_perf)}")
    check(all(v > 0 for v in camp_perf), "campaign_performance all positive", str(camp_perf))

    # Competitor bids — task specific
    comp_bids = obs_dict.get("competitor_bids", [])
    if TASK in BIDDING_TASKS:
        check(len(comp_bids) == 3, "competitor_bids has 3 values (bidding task)", f"got {comp_bids}")
        check(all(v > 0 for v in comp_bids), "competitor_bids all positive", str(comp_bids))
    else:
        print(f"  [INFO] competitor_bids={comp_bids} (budget task — empty expected)", flush=True)

    # Obs history empty at reset
    obs_history = obs_dict.get("obs_history", None)
    check(obs_history == [], "obs_history empty at reset", str(obs_history))

    # Done false at reset
    done = getattr(result, "done", True)
    check(not done, "done=false at reset")

    print(f"  [INFO] vertical in state: {obs_dict.get('vertical', 'N/A')}", flush=True)


# ---------------- TEST 4-5: STEP ----------------
async def test_step(env: AdPlatformClient) -> None:
    section("TEST 4-5 — Step + Task-Specific Checks")

    action = AdPlatformAction(
        allocations=[33.0, 22.0, 11.0],
        bids=[0.6, 0.5, 0.4] if TASK in BIDDING_TASKS else [0.0, 0.0, 0.0],
    )

    try:
        result = await env.step(action)
    except Exception as e:
        failed("step() call", str(e))
        return

    check(result is not None, "step() returned result")

    reward = getattr(result, "reward", None)
    check(reward is not None, "reward present")
    check(
        reward is not None and 0.0 <= reward <= 1.0,
        f"reward in [0, 1]: {reward:.4f}",
    )

    done = getattr(result, "done", None)
    check(done is False, f"done=false after step 1: {done}")

    obs = getattr(result, "observation", result)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else {}

    # Step count
    step_count = obs_dict.get("step", -1)
    check(step_count == 1, f"step=1 after first step", f"got {step_count}")

    # Remaining budget decreased
    total_budget     = obs_dict.get("total_budget", 0)
    remaining_budget = obs_dict.get("remaining_budget", 0)
    check(
        remaining_budget < total_budget,
        f"remaining_budget decreased after spend",
        f"total={total_budget} remaining={remaining_budget}",
    )

    # Reward breakdown
    breakdown = obs_dict.get("reward_breakdown")
    check(breakdown is not None, "reward_breakdown present")
    if breakdown:
        for key in EXPECTED_BREAKDOWN_KEYS:
            check(key in breakdown, f"breakdown has '{key}'")
        gate = breakdown.get("illegal_gate", -1)
        check(gate == 1.0, f"illegal_gate=1.0 (no illegal actions)", f"got {gate}")

    # Obs history has one entry
    obs_history = obs_dict.get("obs_history", [])
    check(len(obs_history) == 1, "obs_history has 1 entry after step 1", f"got {len(obs_history)}")

    # Task-specific — competitor bids
    section("TEST 5 — Task-Specific Checks")
    comp_bids = obs_dict.get("competitor_bids", [])
    if TASK in BIDDING_TASKS:
        check(len(comp_bids) == 3, f"competitor_bids updated after step (task={TASK})", f"got {comp_bids}")
        check(all(v > 0 for v in comp_bids), "competitor_bids all positive after step", str(comp_bids))
        # Bids in obs_history should match what we sent
        sent_bids = obs_history[0].get("bids", []) if obs_history else []
        check(
            len(sent_bids) == 3,
            "bids recorded in obs_history",
            f"got {sent_bids}",
        )
    else:
        check(
            comp_bids == [],
            f"competitor_bids=[] for budget task",
            f"got {comp_bids}",
        )

    print(f"\n  [SUMMARY] task={TASK} reward={reward:.4f} done={done}", flush=True)
    print(f"  [SUMMARY] remaining_budget={remaining_budget:.2f}/{total_budget:.2f}", flush=True)


# ---------------- MAIN ----------------
async def main() -> None:
    print(f"\nAdPlatform Connection Test", flush=True)
    print(f"IMAGE_NAME : {IMAGE_NAME}", flush=True)
    print(f"TASK       : {TASK}", flush=True)
    print(f"VERTICAL   : {VERTICAL or '(none — using defaults)'}", flush=True)

    if not IMAGE_NAME:
        print("\nERROR: IMAGE_NAME not set", flush=True)
        print("Usage: IMAGE_NAME=adplatform_env:latest python test_connection.py", flush=True)
        return

    # Test 1 — market provider (local, before docker)
    test_market_provider()

    # Tests 2-5 — docker connection, reset, step, task checks
    env_vars = {
        "TASK":                       TASK,
        "MARKET_VERTICAL":            VERTICAL or "",
        "MARKET_SAMPLING":            SAMPLING,
        "MARKET_REFRESH_PER_EPISODE": REFRESH,
        "FRED_DATA_API_KEY":           FRED_DATA_API_KEY,
        "YAML_PATH":                  YAML_PATH,
    }
    await test_docker(env_vars)

    print(f"\n{'='*50}", flush=True)
    print(f"  Test run complete", flush=True)
    print(f"{'='*50}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())