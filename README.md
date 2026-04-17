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

Supports synthetic defaults, user-provided campaign YAML, and **real-world market data** sourced automatically from Google Trends, Wikipedia, and FRED economic indicators — making it immediately applicable for training or evaluating bidding and budget agents on realistic market dynamics.

---

## Why This Environment?

Digital advertising platforms (Google Ads, Meta Ads, DSPs) require practitioners to solve exactly these problems every day:

- How much budget to allocate across campaigns this hour/day?
- How high to bid to win auctions without overpaying?
- How to respond to sudden market events (flash sales, competitor surges)?

This environment models those decisions with realistic reward signals, adaptive competitors, real market data per episode, and configurable campaign profiles — making it immediately applicable for training or evaluating bidding and budget agents.

---

## Quick Start

### Option 1 — Run directly (no Docker)

```bash
# Install dependencies
pip install openenv-core numpy openai pyyaml pytrends

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

# Run (auction task with ecommerce market data)
docker run -p 8000:8000 \
  -e TASK=auction \
  -e MARKET_VERTICAL=ecommerce \
  -e HF_TOKEN=<your_api_key> \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ad_platform_env:latest
```

### Option 3 — Connect to this HF Space server

```python
from client import AdPlatformClient
from models import AdPlatformAction

env = await AdPlatformClient.from_docker_image(
    "adplatform_env:latest",
    env_vars={
        "TASK": "auction",
        "MARKET_VERTICAL": "ecommerce",
    }
)
result = await env.reset()
result = await env.step(AdPlatformAction(
    allocations=[30.0, 20.0, 10.0],
    bids=[0.70, 0.55, 0.35]
))
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
- `allocations[i] >= 0` — negative allocations trigger an illegal gate penalty
- Allocations over the per-step pacing limit (30% of remaining budget) are penalized
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
| `obs_history` | `list[dict]` | Last 5 steps: `{step, spend, conversions, competitor_bids, allocations, bids}` |
| `reward` | `float` | Step reward in [0, 1] |
| `done` | `bool` | Episode terminal flag |
| `reward_breakdown` | `dict` | Per-component reward diagnostics |
| `grader_final_score` | `float\|None` | Episode-level grader score (populated at terminal step only) |
| `grader_conversion_score` | `float\|None` | Conversion component of grader score |
| `grader_utilization_score` | `float\|None` | Budget utilization component |
| `grader_bid_efficiency` | `float\|None` | Bid efficiency component (auction/dynamic tasks) |
| `prev_final_score` | `float` | Previous episode's overall grader score (available from step 0) |
| `prev_conversion_score` | `float` | Previous episode's conversion score |
| `prev_utilization_score` | `float` | Previous episode's utilization score |
| `prev_bid_efficiency` | `float` | Previous episode's bid efficiency |
| `prev_episode_graded` | `bool` | Whether a previous episode exists for policy conditioning |

---

## Tasks

### Task 1 — Budget Allocation `[easy]`

**Objective:** Allocate a fixed $1,000 budget across 3 campaigns over 30 steps to maximize total conversions. No competitive bidding — the agent wins all auctions.

**Key challenge:** Pacing the budget evenly while concentrating spend on the highest-converting campaign.

**Grader weights:**
- Conversion score: 70% — `total_conversions / theoretical_max`
- Utilization score: 20% — penalizes leaving budget unspent or overspending early
- Smoothness score: 10% — rewards consistent spend across steps

**Expected agent score:** 0.50–0.75

---

### Task 2 — Competitive Auction `[medium]`

**Objective:** Bid against adaptive competitors while managing a $1,000 budget across 3 campaigns over 30 steps.

**Key challenge:** Win auctions by bidding above competitor bids, but avoid overbidding which wastes budget. Competitors adapt to the agent's previous bids with a responsiveness factor `alpha=0.3`.

**Win probability:** `P(win) = sigmoid(agent_bid - competitor_bid)`

**Grader weights:**
- Conversion score: 60%
- Utilization score: 25%
- Bid efficiency: 15% — conversions per $ spent, normalized

**Expected agent score:** 0.30–0.55

---

### Task 3 — Dynamic Campaign Management `[hard]`

**Objective:** Full campaign management with seasonality, market events (flash sales / demand spikes), adaptive competitors, and per-step conversion rate shifts.

**Key challenge:** Detect and exploit market events. Adapt spending and bids to capitalize on temporary opportunities while maintaining overall budget pacing.

**Grader weights:**
- Conversion score: 50%
- Utilization score: 25%
- Bid efficiency: 15%
- Adaptability: 10% — did the agent spend more during market event steps?

**Expected agent score:** 0.20–0.45

---

## Reward Function

Each step returns a shaped reward in [0, 1] providing partial progress signals throughout the episode.

**Components (shared across tasks):**

| Component | Role |
|---|---|
| Conversion signal | Normalized conversions earned this step |
| Illegal gate | Multiplicative suppressor when negative/overlimit allocations occur |
| Spend penalty | Discourages spending the pacing limit in one step |
| Carryover penalty | Penalizes aggressive early spending |
| Terminal bonus | One-time bonus at episode end for strong trajectory-level performance |

**Task-specific additions:**
- Task 2/3 add **bid quality signal**: rewards winning auctions on high-conversion campaigns
- Task 2 adds **competitor-aware pacing**: penalizes spending when likely to lose auctions
- Task 3 adds **adaptability signal**: rewards spending more during market event steps

**Dynamic reward bounds:** `MAX_CONV_PER_STEP` and `MAX_ILLEGAL_PENALTY` are computed from the actual profile in use each episode — so normalization stays correct regardless of `total_budget`, `conversion_rates`, or `seasonal_multipliers` provided via `CampaignProfile` or `MarketDataProvider`.

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

## Profile Data Hierarchy

The environment resolves campaign parameters from multiple sources, lowest to highest priority:

```
Tier 1 — Synthetic defaults       AdPlatformState field defaults (always present)
Tier 2 — default_profile.yaml     Loaded automatically as safeguard when no yaml_path set
Tier 3 — User YAML (yaml_path)    Your own Google Ads / Meta Ads export
Tier 4 — MarketDataProvider       Real market data per episode (via MARKET_VERTICAL)
Tier 5 — Runtime CampaignProfile  Passed to reset() per episode — highest priority
```

Each tier only overrides fields it explicitly provides — anything absent falls through to the tier below.

---

## CampaignProfile — Injecting Real Data

`CampaignProfile` is a validated dict that lets you replace synthetic defaults with real historical campaign data. It can be passed at reset time, loaded from a YAML file, or both.

### Fields

| Field | Type | Description |
|---|---|---|
| `conversion_rates` | `list[float]` | CVR per campaign (conversions per $ spent). Must be ≥ 0. |
| `competitor_bids` | `list[float]` | Baseline competitor bid per campaign ($). Must be > 0. |
| `bid_volatility` | `list[float]` | Per-campaign bid noise std-dev fraction (e.g., 0.10 = ±10%). |
| `seasonal_multipliers` | `list[float]` | One multiplier per step; wraps around if shorter than max_steps. |
| `market_events` | `dict[int, list[float]]` | Step → per-campaign CVR multiplier for known sale/demand events. |
| `total_budget` | `float` | Total episode budget ($). Must be > 0. |

All fields are **optional** — anything omitted falls back to the next tier in the hierarchy.

**Shape rule:** `conversion_rates`, `competitor_bids`, and `bid_volatility` must all have length 3 if provided together.

### Method 1 — Pass directly to `reset()`

```python
from models import CampaignProfile
from server.environment import AdPlatformEnvironment

env = AdPlatformEnvironment(task="auction")

profile = CampaignProfile(
    conversion_rates=[0.04, 0.02, 0.015],
    competitor_bids=[0.80, 0.60, 0.45],
    bid_volatility=[0.08, 0.12, 0.06],
    seasonal_multipliers=[1.3, 1.0, 0.8, 1.2, 1.4, 1.5, 0.8],
    market_events={5: [1.4, 1.0, 0.9], 18: [1.0, 1.5, 1.0]},
    total_budget=5000.0,
)

obs = env.reset(profile=profile)
```

### Method 2 — Load from YAML file

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

```python
env = AdPlatformEnvironment(task="dynamic_campaign", yaml_path="my_ads_export.yaml")
obs = env.reset()
```

Or via environment variable:

```bash
YAML_PATH=my_ads_export.yaml uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Method 3 — Merge YAML base with per-episode override

```python
env = AdPlatformEnvironment(task="auction", yaml_path="my_ads_export.yaml")

# Only override budget for this episode — all other fields come from YAML
episode_profile = CampaignProfile(total_budget=3000.0)
obs = env.reset(profile=episode_profile)
```

### CampaignProfile validation

```python
# Unknown field
CampaignProfile(cpc=[0.5, 0.4, 0.3])
# → ValueError: CampaignProfile received unknown fields: {'cpc'}

# Shape mismatch
CampaignProfile(conversion_rates=[0.04, 0.02], competitor_bids=[0.8, 0.6, 0.45])
# → ValueError: Per-campaign fields must all have the same length. Got: {'conversion_rates': 2, 'competitor_bids': 3}

# Negative value
CampaignProfile(conversion_rates=[-0.01, 0.03, 0.02])
# → ValueError: conversion_rates must contain non-negative numbers
```

---

## MarketDataProvider — Real-World Market Data

`MarketDataProvider` sources real advertising market dynamics and converts them into `CampaignProfile` objects — one per episode. This gives every episode a different but historically grounded market context without requiring your own ad account data.

### Data Sources (priority chain)

```
1. Google Trends (live)    Real search interest → seasonal patterns, competition levels
2. Google Trends (cached)  On-disk cache, max 7 days old
3. Wikipedia (live)        Pageview API — no auth, highly reliable fallback
4. Wikipedia (cached)      On-disk cache, max 7 days old
5. FRED (live)             Macro economic indicators — vertical-specific
6. FRED (cached)           On-disk cache, max 7 days old
7. default_profile.yaml    Always works offline — last resort
```

### Verticals

| Vertical | Keywords | FRED Series | Budget |
|---|---|---|---|
| `ecommerce` | nike / running shoes / online shopping | RSXFS (retail sales) | $1,000 |
| `saas` | salesforce / crm software / business software | DGORDER (durable goods) | $2,000 |
| `travel` | expedia / cheap flights / travel deals | TOTALSA (vehicle sales) | $1,500 |
| `finance` | chase bank / personal loan / financial services | UMCSENT (consumer sentiment) | $3,000 |

### Three Episode Sampling Modes

| Mode | Behaviour | Use Case |
|---|---|---|
| Fixed (`refresh_per_episode=False`) | Same 30-day window every episode | Debugging, reproducibility |
| Sequential (`sampling="sequential"`) | Slides 7 days forward each episode, wraps at end | Realistic market progression — default |
| Random (`sampling="random"`) | Random 30-day window each episode | Maximum training diversity |

### What it populates per episode

From the selected 30-day window of real data:

- `conversion_rates` — base benchmark CVR scaled by market competition index (sqrt dampening)
- `competitor_bids` — base benchmark CPC scaled linearly by search interest level
- `bid_volatility` — coefficient of variation of the trend window
- `seasonal_multipliers` — 30 real data points averaged across all three campaign signals
- `market_events` — automatically detected from z-score spikes (top 3 most anomalous steps)
- `total_budget` — industry benchmark from WordStream per vertical

### Usage

**Via environment variable (server/Docker):**

```bash
# Start server with ecommerce market data
MARKET_VERTICAL=ecommerce uvicorn server.app:app --port 8000

# Docker
docker run -p 8000:8000 \
  -e TASK=auction \
  -e MARKET_VERTICAL=ecommerce \
  -e MARKET_SAMPLING=sequential \
  -e MARKET_REFRESH_PER_EPISODE=true \
  ad_platform_env:latest
```

**Via `vertical` parameter (local):**

```python
from server.environment import AdPlatformEnvironment

# Vertical set at init — used for all episodes
env = AdPlatformEnvironment(task="auction", vertical="ecommerce")
obs = env.reset()

# Switch vertical at reset — creates/switches MarketDataProvider
obs = env.reset(vertical="saas")

# Stay on last set vertical
obs = env.reset()
```

**Direct usage:**

```python
from data_build.market_data_provider import MarketDataProvider

provider = MarketDataProvider(
    vertical="ecommerce",
    refresh_per_episode=True,
    sampling="sequential",
)
print(f"Source: {provider.source}")          # Google Trends / Wikipedia / FRED
print(f"Dates: {provider.data_date_range}")  # e.g. 2026-01-12 to 2026-04-11

profile = provider.get_profile()  # CampaignProfile ready for reset()
```

### Optional — FRED API Key

```bash
# Free key at https://fred.stlouisfed.org/docs/api/api_key.html
# Improves FRED data quality — without it public endpoint is used (limited)
export FRED_DATA_API_KEY=your_key
```

### Vertical Benchmarks

Industry benchmarks are stored in `data_build/vertical_benchmarks.yaml` — update quarterly as WordStream publishes new data. No code changes needed.

---

## Environment Variables Reference

### Mandatory (for inference)

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |

### Optional — Task & Data

| Variable | Default | Description |
|---|---|---|
| `TASK` | `budget` | Task name: `budget` \| `auction` \| `dynamic_campaign` |
| `YAML_PATH` | — | Path to your own campaign profile YAML |
| `IMAGE_NAME` | — | Docker image name (if using `from_docker_image`) |
| `BENCHMARK` | `ad_platform_env` | Benchmark label in logs |

### Optional — Market Data

| Variable | Default | Description |
|---|---|---|
| `MARKET_VERTICAL` | — | `ecommerce` \| `saas` \| `travel` \| `finance`. If not set — uses `default_profile.yaml` |
| `MARKET_SAMPLING` | `sequential` | `sequential` \| `random` |
| `MARKET_REFRESH_PER_EPISODE` | `true` | Whether to slide the market data window each episode |
| `FRED_DATA_API_KEY` | — | FRED API key for improved macro data quality |

---

## OpenEnv API

```python
from server.environment import AdPlatformEnvironment
from models import AdPlatformAction, CampaignProfile

# Local — no Docker
env = AdPlatformEnvironment(
    task="auction",
    yaml_path="my_ads_export.yaml",   # optional
    vertical="ecommerce",              # optional — real market data
)

# reset() — starts a new episode
obs = env.reset()

# reset() with runtime overrides
obs = env.reset(
    task="dynamic_campaign",   # switch task
    vertical="saas",           # switch vertical
    profile=CampaignProfile(total_budget=3000.0),  # episode-level override
)

# step(action) → StepResult with observation, reward, done
result = env.step(AdPlatformAction(
    allocations=[30.0, 20.0, 10.0],
    bids=[0.70, 0.55, 0.35],
))
print(result.reward)   # float in [0, 1]
print(result.done)     # bool
print(result.observation.remaining_budget)
print(result.observation.reward_breakdown)

# state → current episode state
s = env.state
print(s.step_count, s.total_conversions, s.remaining_budget)
```

### HTTP API (when running the server)

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take a step, returns observation + reward |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/grade/budget` | GET | Episode grader score for budget task |
| `/grade/auction` | GET | Episode grader score for auction task |
| `/grade/dynamic_campaign` | GET | Episode grader score for dynamic campaign task |
| `/docs` | GET | OpenAPI/Swagger documentation |

---

## Inference Script

`inference.py` runs a model-driven agent against all three tasks sequentially using the OpenAI client interface.

```bash
export HF_TOKEN=<your_api_key>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export MARKET_VERTICAL=ecommerce   # optional — real market data

python inference.py
```

### Log Format

```
[START] task=budget env=ad_platform_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={'allocations': [30.0, 20.0, 10.0], 'bids': [0.0, 0.0, 0.0]} reward=0.42 done=false error=null conv=0.312 bid=0.000 pacing=0.175 gate=1.000
...
[END] success=false steps=30 score=0.421 rewards=0.42,0.38,...
[START] task=auction env=ad_platform_env model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=30 score=0.512 rewards=...
[START] task=dynamic_campaign ...
...
[END] ...
```

**Score:** `sum(step_rewards) / MAX_STEPS` — mean normalized reward per step, in [0, 1].
**Success threshold:** score ≥ 0.5

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HF Inference API, 30 steps per episode:

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| `budget` | Easy | ~0.55–0.65 | Model learns allocation quickly |
| `auction` | Medium | ~0.35–0.50 | Adaptive bidding requires multi-step reasoning |
| `dynamic_campaign` | Hard | ~0.25–0.45 | Market event exploitation is challenging |

---

## Project Structure

```
ad_platform_env/
├── data_build/
│   ├── __init__.py
│   ├── market_constants.py           # Keyword mappings and tuning constants
│   ├── market_data_provider.py       # ETL — Google Trends / Wikipedia / FRED
│   └── vertical_benchmarks.yaml      # Industry benchmarks per vertical (update quarterly)
├── server/
│   ├── grader/                       # Episode-level graders
│   ├── tasks/                        # Reset and step functions per task
│   ├── __init__.py
│   ├── app.py                        # FastAPI app (OpenEnv HTTP server)
│   ├── default_profile.yaml          # Safeguard profile (mid-sized Google Ads account)
│   ├── environment.py                # AdPlatformEnvironment (reset/step/state)
│   ├── profile_loader.py             # YAML profile loading & merging
│   └── requirements.txt
├── tests/
│   ├── test_connection_v1.py
│   └── test_connection.py            # End-to-end connection test
├── __init__.py
├── .dockerignore
├── .gitignore
├── .python-version
├── client.py                         # AdPlatformClient (WebSocket / Docker)
├── Dockerfile
├── env
├── http_server.py
├── inference_final.py
├── inference.py                      # Multi-task inference script
├── models.py                         # AdPlatformAction, AdPlatformObservation,
│                                     #   AdPlatformState, CampaignProfile
├── openenv.yaml                      # OpenEnv manifest
├── pyproject.toml
├── README.md
├── uv.lock
└── validate-submission.sh
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Hugging Face account + token (for inference)
- SSL certificates installed (macOS: run `/Applications/Python 3.x/Install Certificates.command`)

### Local Setup

```bash
git clone <repo-url>
cd ad_platform_env
pip install -e ".[dev]"

# Start server
uvicorn server.app:app --reload --port 8000

# Validate
openenv validate
```

### Docker Build & Run

```bash
# Build
docker build -t ad_platform_env:latest .

# Run — auction task with ecommerce market data
docker run -p 8000:8000 \
  -e TASK=auction \
  -e MARKET_VERTICAL=ecommerce \
  ad_platform_env:latest

# Run — with your own YAML profile
docker run -p 8000:8000 \
  -e TASK=dynamic_campaign \
  -e YAML_PATH=/data/my_export.yaml \
  -v /path/to/your/data:/data \
  ad_platform_env:latest
```

### Deploy to Hugging Face Spaces

```bash
huggingface-cli login
openenv push
# or
openenv push --repo-id my-org/ad-platform-env --private
```

---

## Pre-Submission Validation

```bash
bash validate-submission.sh
```

Checks:
- `openenv validate` passes
- `docker build` succeeds
- All 3 tasks respond to `reset()` and `step()`
- Grader scores are in [0.0, 1.0]
- Inference script runs and produces valid logs

---

## Sourcing Real Campaign Data

Export your campaign data from Google Ads, Meta Ads, or any DSP into the YAML format:

```yaml
# my_google_ads_export.yaml
conversion_rates: [0.042, 0.019, 0.011]   # from conversion tracking
competitor_bids: [0.78, 0.63, 0.41]       # average CPC from Auction Insights
bid_volatility: [0.06, 0.14, 0.08]        # std-dev from bid history
seasonal_multipliers: [1.10, 1.15, 1.15, 1.10, 1.05, 0.85, 0.80]  # Mon–Sun
market_events:
  7: [1.5, 1.0, 0.9]    # flash sale
  21: [1.0, 1.4, 1.1]   # competitor promotion
total_budget: 5000.0
```

Pass it at startup:

```bash
YAML_PATH=my_google_ads_export.yaml python inference.py
```

Or inject per episode:

```python
profile = CampaignProfile(
    conversion_rates=[0.042, 0.019, 0.011],
    total_budget=3000.0,
)
obs = env.reset(task="dynamic_campaign", profile=profile)
```

If you don't have your own data — set `MARKET_VERTICAL=ecommerce` (or another vertical) and the environment will source real market dynamics automatically from public data sources.
