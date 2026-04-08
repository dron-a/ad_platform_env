---
title: Ad Platform Env Environment Server
emoji: 📊
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ad Platform RL Environment

A real-world OpenEnv environment that trains and evaluates AI agents on digital advertising tasks — budget allocation, competitive auction bidding, and dynamic campaign management with seasonality and market events.

---

## Why This Environment?

Digital advertising platforms (Google Ads, Meta Ads, DSPs) require practitioners to solve exactly these problems every day:

- How much budget to allocate across campaigns this hour/day?
- How high to bid to win auctions without overpaying?
- How to respond to sudden market events (flash sales, competitor surges)?

This environment models those decisions with realistic reward signals, adaptive competitors, and configurable real-world campaign data — making it immediately applicable for training or evaluating bidding/budget agents.

---

## Quick Start

### Option 1 — Run directly (no Docker)

```bash
# Install dependencies
pip install openenv-core numpy openai pyyaml

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal — run the baseline inference agent
export HF_TOKEN=<your_api_key>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export TASK=auction

python inference.py
```

### Option 2 — Docker

```bash
# Build
docker build -t ad_platform_env:latest .

# Run (budget task by default)
docker run -p 8000:8000 \
  -e TASK=auction \
  -e HF_TOKEN=<your_api_key> \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ad_platform_env:latest
```

---

## Environment Description

The Ad Platform environment simulates an automated bidding agent managing three ad campaigns over a 30-step episode. Each step represents one decision cycle (e.g., one hour of ad spend).

**Campaigns:**
- Campaign 0: Branded search — high intent, best conversion rate
- Campaign 1: Generic search — medium intent, moderate conversion rate
- Campaign 2: Display / retargeting — broad reach, lowest conversion rate

The agent must allocate budget and set bids across campaigns each step to maximize total conversions while respecting a fixed episode budget.

---

## Action Space

**Type:** `AdPlatformAction` (Pydantic model)

| Field | Type | Description |
|---|---|---|
| `allocations` | `list[float]` | Budget to spend per campaign this step (3 values, non-negative) |
| `bids` | `list[float]` | Bid price per campaign (3 values; required for auction/dynamic tasks, ignored for budget task) |

**Example action:**
```json
{
  "allocations": [30.0, 20.0, 10.0],
  "bids": [0.70, 0.55, 0.35]
}
```

**Constraints:**
- `allocations[i] >= 0` — negative allocations trigger a penalty
- Allocations over the per-step pacing limit (30% of remaining budget) are clamped and penalized
- Budget is shared across all campaigns — total spend cannot exceed `remaining_budget`

---

## Observation Space

**Type:** `AdPlatformObservation` (Pydantic model)

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current timestep (0–29) |
| `total_budget` | `float` | Total episode budget ($) |
| `remaining_budget` | `float` | Budget not yet spent |
| `campaign_performance` | `list[float]` | Conversion rate per campaign (conversions per $ spent) |
| `competitor_bids` | `list[float]` | Current competitor bid per campaign (auction/dynamic tasks) |
| `obs_history` | `list[dict]` | Last 5 steps of context: `{step, spend, conversions, competitor_bids, allocations, bids}` |
| `reward` | `float` | Step reward in [0, 1] |
| `done` | `bool` | Episode terminal flag |
| `reward_breakdown` | `dict` | Per-component reward diagnostics |
| `grader_final_score` | `float\|None` | Episode-level grader score (populated at terminal step only) |
| `grader_conversion_score` | `float\|None` | Conversion component of grader score |
| `grader_utilization_score` | `float\|None` | Budget utilization component |
| `grader_bid_efficiency` | `float\|None` | Bid efficiency component (auction/dynamic tasks) |
| `prev_final_score` | `float` | Previous episode's grader score (available from step 0 for policy conditioning) |
| `prev_conversion_score` | `float` | Previous episode's conversion score |
| `prev_utilization_score` | `float` | Previous episode's utilization score |
| `prev_bid_efficiency` | `float` | Previous episode's bid efficiency |
| `prev_episode_graded` | `bool` | Whether a previous episode exists |

---

## Tasks

### Task 1 — Budget Allocation `[easy]`

**Objective:** Allocate a fixed $1,000 budget across 3 campaigns over 30 steps to maximize total conversions. No competitive bidding — the agent wins all auctions.

**Key challenge:** Pacing the budget evenly while concentrating spend on the highest-converting campaign.

**Grader weights:**
- Conversion score: 70% — `total_conversions / theoretical_max`
- Utilization score: 20% — penalizes leaving budget unspent or overspending early
- Smoothness score: 10% — rewards consistent spend across steps

**Expected agent score:** 0.50–0.75 (straightforward once the agent learns the conversion rates)

---

### Task 2 — Competitive Auction `[medium]`

**Objective:** Bid against adaptive competitors while managing a $1,000 budget across 3 campaigns over 30 steps.

**Key challenge:** Win auctions by bidding above competitor bids (win probability via sigmoid function), but avoid overbidding which wastes budget and reduces efficiency. Competitors adapt to the agent's previous bids.

**Win probability:** `P(win) = sigmoid(agent_bid - competitor_bid)`

**Grader weights:**
- Conversion score: 60%
- Utilization score: 25%
- Bid efficiency: 15% — conversions per $ spent, normalized

**Expected agent score:** 0.30–0.55 (requires learning adaptive bidding strategies)

---

### Task 3 — Dynamic Campaign Management `[hard]`

**Objective:** Full campaign management with seasonality, market events (flash sales / demand spikes), adaptive competitors, and per-step conversion rate shifts.

**Key challenge:** Detect and exploit market events (e.g., step 10: Campaign 0 gets 20% CVR boost; step 20: Campaign 1 gets 30% boost). Adapt spending and bids to capitalize on temporary opportunities while maintaining overall budget pacing.

**Market event defaults:**
- Step 10: `[1.2, 1.0, 0.8]` — Campaign 0 CVR boost
- Step 20: `[0.9, 1.3, 1.0]` — Campaign 1 CVR boost

**Grader weights:**
- Conversion score: 50%
- Utilization score: 25%
- Bid efficiency: 15%
- Adaptability: 10% — did the agent spend more during market event steps?

**Expected agent score:** 0.20–0.45 (requires dynamic strategy adjustment)

---

## Reward Function

Each step returns a shaped reward in [0, 1] providing partial progress signals throughout the episode.

**Components (shared across tasks):**

| Component | Role |
|---|---|
| Conversion signal | Normalized conversions earned this step |
| Illegal gate | Multiplicative suppressor when negative/overlimit allocations occur |
| Spend penalty | Discourages spending the pacing limit in one step |
| Carryover penalty | Penalizes aggressive early spending (leaves nothing for later steps) |
| Terminal bonus | One-time bonus at episode end for strong trajectory-level performance |

**Task-specific additions:**
- Task 2/3 add **bid quality signal**: rewards winning auctions on high-conversion campaigns
- Task 2 adds **competitor-aware pacing**: penalizes spending when likely to lose auctions
- Task 3 adds **adaptability signal**: rewards spending more during market event steps

**Reward breakdown** is included in every observation for diagnostics:
```json
{
  "conv_component": 0.312,
  "bid_quality_component": 0.201,
  "pacing_component": 0.175,
  "illegal_gate": 1.0,
  "step_reward": 0.487
}
```

---

## CampaignProfile — Injecting Real Data

`CampaignProfile` is a validated dict that lets you replace the synthetic defaults with real historical campaign data from your ad platform export. It can be passed at reset time, loaded from a YAML file, or both.

### Fields

| Field | Type | Description |
|---|---|---|
| `conversion_rates` | `list[float]` | CVR per campaign (conversions per $ spent). Must be ≥ 0. |
| `competitor_bids` | `list[float]` | Baseline competitor bid per campaign ($). Must be > 0. |
| `bid_volatility` | `list[float]` | Per-campaign bid noise std-dev fraction (e.g., 0.10 = ±10%). |
| `seasonal_multipliers` | `list[float]` | One multiplier per step; wraps around if shorter than max_steps. |
| `market_events` | `dict[int, list[float]]` | Step number → per-campaign CVR multiplier for known sale events. |
| `total_budget` | `float` | Total episode budget ($). Must be > 0. |

All fields are **optional** — anything you omit falls back to the environment's built-in defaults.

**Shape rule:** `conversion_rates`, `competitor_bids`, and `bid_volatility` must all have length 3 (one per campaign) if provided together.

### Method 1 — Pass directly to `reset()`

```python
from models import CampaignProfile
from server.environment import AdPlatformEnvironment

env = AdPlatformEnvironment(task="auction")

# Build from your Google Ads / Meta Ads export
profile = CampaignProfile(
    conversion_rates=[0.04, 0.02, 0.015],       # real historical CVR
    competitor_bids=[0.80, 0.60, 0.45],          # average CPC competitors paid
    bid_volatility=[0.08, 0.12, 0.06],           # std-dev of competitor bid swings
    seasonal_multipliers=[1.3, 1.0, 0.8, 1.2, 1.4, 1.5, 0.8],  # weekly pattern
    market_events={5: [1.4, 1.0, 0.9], 18: [1.0, 1.5, 1.0]},   # known sale events
    total_budget=5000.0,
)

obs = env.reset(task="auction", profile=profile)
```

### Method 2 — Load from YAML file

Create a YAML file (e.g., `my_ads_export.yaml`):

```yaml
# my_ads_export.yaml
conversion_rates: [0.04, 0.02, 0.015]
competitor_bids: [0.80, 0.60, 0.45]
bid_volatility: [0.08, 0.12, 0.06]
seasonal_multipliers: [1.3, 1.1, 0.9, 1.2, 1.4, 1.5, 0.8]
market_events:
  3: [1.5, 1.0, 0.9]
  15: [1.0, 1.3, 1.0]
total_budget: 5000.0
```

Pass the path at environment startup:

```python
env = AdPlatformEnvironment(task="dynamic_campaign", yaml_path="my_ads_export.yaml")
obs = env.reset()  # automatically uses YAML profile
```

Or set via environment variable when running the server:

```bash
YAML_PATH=my_ads_export.yaml uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Method 3 — Merge YAML base with per-episode override

The three-tier priority system lets you set a YAML baseline and override specific fields per episode:

```python
from models import CampaignProfile
from server.profile_loader import ProfileLoader
from server.environment import AdPlatformEnvironment

# Tier 2: YAML file loaded at startup
loader = ProfileLoader("my_ads_export.yaml")

env = AdPlatformEnvironment(task="auction")

# Tier 3: runtime override for this specific episode
episode_profile = CampaignProfile(total_budget=3000.0)  # only override budget

# Merged: YAML base + episode override (runtime wins)
merged = ProfileLoader.resolve(loader.profile, episode_profile)
obs = env.reset(profile=merged)
```

### Data-tier priority (lowest → highest)

```
Tier 1: Built-in synthetic defaults  (always present)
Tier 2: YAML file (yaml_path=...)    (overrides Tier 1 for present fields)
Tier 3: reset(profile=...)           (overrides Tiers 1 & 2 for present fields)
```

### CampaignProfile validation

The class validates inputs and raises descriptive errors:

```python
# Wrong type
CampaignProfile(conversion_rates="0.04")
# → TypeError: conversion_rates must be a list

# Negative value
CampaignProfile(conversion_rates=[-0.01, 0.03, 0.02])
# → ValueError: conversion_rates must contain non-negative numbers

# Shape mismatch
CampaignProfile(conversion_rates=[0.04, 0.02], competitor_bids=[0.8, 0.6, 0.45])
# → ValueError: Per-campaign fields must all have the same length. Got: {'conversion_rates': 2, 'competitor_bids': 3}

# Unknown field
CampaignProfile(cpc=[0.5, 0.4, 0.3])
# → ValueError: CampaignProfile received unknown fields: {'cpc'}. Valid fields are: {...}
```

---

## OpenEnv API

The environment implements the full OpenEnv spec:

```python
from server.environment import AdPlatformEnvironment
from models import AdPlatformAction

env = AdPlatformEnvironment(task="auction")

# reset() → AdPlatformObservation
obs = env.reset()

# step(action) → AdPlatformObservation with reward + done
result = env.step(AdPlatformAction(
    allocations=[30.0, 20.0, 10.0],
    bids=[0.70, 0.55, 0.35]
))
print(result.reward, result.done)

# state → State
s = env.state
print(s.remaining_budget, s.total_conversions)
```

### HTTP API (when running the server)

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take a step, returns observation + reward |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI/Swagger documentation |
| `/web` | GET | Interactive web UI |

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Hugging Face account + token (for inference against HF-hosted models)

### Local Setup

```bash
# Clone and install
git clone <repo-url>
cd ad_platform_env
pip install -e ".[dev]"

# Run server
uvicorn server.app:app --reload --port 8000

# Validate OpenEnv spec compliance
openenv validate
```

### Required Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | Hugging Face API key |
| `TASK` | No | `auction` | Task name: `budget`, `auction`, or `dynamic_campaign` |
| `YAML_PATH` | No | — | Path to campaign profile YAML |
| `IMAGE_NAME` | No | — | Docker image name (if using `from_docker_image`) |
| `BENCHMARK` | No | `ad_platform_env` | Benchmark label in logs |

### Docker Build & Run

```bash
# Build image
docker build -t ad_platform_env:latest .

# Run server (budget task)
docker run -p 8000:8000 -e TASK=budget ad_platform_env:latest

# Run server (auction task) with your own YAML profile
docker run -p 8000:8000 \
  -e TASK=auction \
  -e YAML_PATH=/data/my_export.yaml \
  -v /path/to/your/data:/data \
  ad_platform_env:latest

# Run inference script inside container
docker run \
  -e TASK=auction \
  -e HF_TOKEN=<your_token> \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ad_platform_env:latest \
  python inference.py
```

### Deploy to Hugging Face Spaces

```bash
# Authenticate
huggingface-cli login

# Push via openenv CLI
openenv push

# Or push to a specific repo
openenv push --repo-id my-org/ad-platform-env

# Push as private space
openenv push --private
```

After deployment, the space is available at:
`https://huggingface.co/spaces/<username>/ad_platform_env`

---

## Inference Script

`inference.py` runs a model-driven agent against the environment using the OpenAI client interface. It emits structured logs required for hackathon evaluation.

```bash
# Set required variables
export HF_TOKEN=<your_api_key>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export TASK=auction            # budget | auction | dynamic_campaign

# Run
python inference.py
```

### Log Format

The script emits structured stdout logs:

```
[START] task=auction env=ad_platform_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={'allocations': [30.0, 20.0, 10.0], 'bids': [0.70, 0.55, 0.35]} reward=0.42 done=false error=null conv=0.312 bid=0.201 pacing=0.175 gate=1.000
[STEP] step=2 action=... reward=0.38 done=false error=null ...
...
[END] success=true steps=30 score=0.421 rewards=0.42,0.38,...
```

**Score:** `sum(step_rewards) / MAX_STEPS` — mean normalized reward per step, in [0, 1].

**Success threshold:** score ≥ 0.5

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HF Inference API, 30 steps per episode:

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| `budget` | Easy | ~0.55–0.65 | Model learns allocation quickly |
| `auction` | Medium | ~0.35–0.50 | Bidding strategy requires multi-step reasoning |
| `dynamic_campaign` | Hard | ~0.25–0.40 | Market event exploitation is challenging |

> Scores depend on model temperature, API latency, and market conditions. Re-run with `seed=42` for reproducibility.

---

## Project Structure

```
ad_platform_env/
├── README.md                        # This file
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Project metadata and dependencies
├── Dockerfile                       # Container image definition
├── inference.py                     # Baseline inference script (required)
├── models.py                        # AdPlatformAction, AdPlatformObservation,
│                                    #   AdPlatformState, CampaignProfile
├── client.py                        # AdPlatformClient (HTTP / Docker)
├── __init__.py                      # Module exports
├── validate-submission.sh           # Pre-submission validation script
└── server/
    ├── app.py                       # FastAPI app (OpenEnv HTTP server)
    ├── environment.py               # AdPlatformEnvironment (reset/step/state)
    ├── profile_loader.py            # YAML profile loading & merging
    ├── default_profile.yaml         # Default mid-sized Google Ads profile
    ├── requirements.txt             # Server dependencies
    ├── tasks/
    │   ├── task1_budget.py          # Task 1: budget allocation
    │   ├── task2_auction.py         # Task 2: competitive auction
    │   └── task3_dynamic_campaign.py # Task 3: dynamic campaign management
    └── grader/
        ├── episode_grader.py        # Episode-level graders (compute_score, etc.)
        ├── reward_base.py           # Shared reward components & constants
        ├── reward_task1_budget.py   # Task 1 step reward
        ├── reward_task2_auction.py  # Task 2 step reward
        └── reward_task3_dyn.py      # Task 3 step reward
```

---

## Pre-Submission Validation

Run the provided validation script before submitting:

```bash
bash validate-submission.sh
```

This checks:
- `openenv validate` passes
- `docker build` succeeds
- All 3 tasks respond to `reset()` and `step()`
- Grader scores are in [0.0, 1.0]
- Inference script runs and produces valid logs

---

## Environment Variables for Custom Campaign Data

To use your own ad platform export rather than the synthetic defaults:

1. Export your campaign data (Google Ads, Meta Ads, etc.) into the YAML format:

```yaml
# my_google_ads_export.yaml
conversion_rates: [0.042, 0.019, 0.011]   # from Ads conversion tracking
competitor_bids: [0.78, 0.63, 0.41]       # average auction CPC from Auction Insights
bid_volatility: [0.06, 0.14, 0.08]        # std-dev from bid history
seasonal_multipliers: [1.10, 1.15, 1.15, 1.10, 1.05, 0.85, 0.80]  # Mon–Sun
market_events:
  7: [1.5, 1.0, 0.9]    # flash sale day
  21: [1.0, 1.4, 1.1]   # competitor promotion
total_budget: 5000.0
```

2. Pass it when starting the server:

```bash
YAML_PATH=my_google_ads_export.yaml uvicorn server.app:app --port 8000
# or
YAML_PATH=my_google_ads_export.yaml python inference.py
```

3. Or inject per episode:

```python
from models import CampaignProfile
profile = CampaignProfile(total_budget=3000.0, conversion_rates=[0.042, 0.019, 0.011])
obs = env.reset(task="dynamic_campaign", profile=profile)
```
