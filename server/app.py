# server/app.py
import os
import argparse
import uvicorn
from fastapi.responses import HTMLResponse
from pathlib import Path
from server.grader import compute_score, compute_auction_score, compute_dynamics_campaign_score

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AdPlatformAction, AdPlatformObservation
    from .environment import AdPlatformEnvironment
except ImportError:
    from models import AdPlatformAction, AdPlatformObservation
    from server.environment import AdPlatformEnvironment

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Pre-fetch market data at startup in main thread
# Avoids network calls inside executor thread which can timeout
_MARKET_PROVIDER = None
_vertical = os.getenv("MARKET_VERTICAL", None)
if _vertical:
    try:
        from ..data_build import MarketDataProvider
    except ImportError:
        from data_build import MarketDataProvider
    try:
        _MARKET_PROVIDER = MarketDataProvider(
            vertical=_vertical,
            refresh_per_episode=os.getenv("MARKET_REFRESH_PER_EPISODE", "true").lower() == "true",
            sampling=os.getenv("MARKET_SAMPLING", "sequential"),
        )
        print(
            f"[APP] Market data pre-fetched — "
            f"vertical={_MARKET_PROVIDER.vertical} "
            f"source={_MARKET_PROVIDER.source} "
            f"dates={_MARKET_PROVIDER.data_date_range}",
            flush=True,
        )
    except Exception as e:
        print(f"[APP] Market data pre-fetch failed: {e} — using defaults", flush=True)


# This dictionary lives as long as the server is running
# It maps Session IDs -> Environment Instances
_USER_ENVIRONMENTS = {}

######## Multiple user version of get_Ad_platform_env ##############################################################################
# to be uused in place of the below function if need to make server for multiple users trainig their model using their own instance

# # Per session — new instance but uses class-level cache
# def get_ad_platform_env(session_id="default"):
#     try:
#         task_name = os.getenv("TASK", "budget")
#         yaml_path = os.getenv("YAML_PATH", None) or None
#         vertical  = os.getenv("MARKET_VERTICAL", None)

#         if session_id not in _USER_ENVIRONMENTS:
#             print(
#                     f"[APP] New session={session_id} task={task_name} "
#                     f"vertical={vertical or 'none'}",
#                     flush=True,
#                 )
#             session_provider = None
#             if _MARKET_PROVIDER is not None:
#                 # Create per-session provider sharing the pre-fetched dataset
#                 try:
#                     from ..data_build import MarketDataProvider
#                 except ImportError:
#                     from data_build import MarketDataProvider
#                 session_provider = MarketDataProvider(
#                     vertical=_MARKET_PROVIDER.vertical,
#                     refresh_per_episode=_MARKET_PROVIDER.refresh_per_episode,
#                     sampling=_MARKET_PROVIDER.sampling,
#                 )
#                 # No network fetch — class-level cache already populated
            
#             _USER_ENVIRONMENTS[session_id] = AdPlatformEnvironment(
#                 task=task_name,
#                 yaml_path=yaml_path,
#                 vertical=vertical,        # preserve vertical identity
#                 _market_provider=session_provider,  # pass pre-fetched data
#             )

#         return _USER_ENVIRONMENTS[session_id]

#     except Exception as e:
#         import traceback
#         print(f"[FACTORY] EXCEPTION: {e}", flush=True)
#         traceback.print_exc()
#         raise

def get_ad_platform_env(session_id: str = "default"):
    print(f"[FACTORY] called session={session_id} _MARKET_PROVIDER={_MARKET_PROVIDER}", flush=True)
    try:
        task_name = os.getenv("TASK", "budget")
        yaml_path = os.getenv("YAML_PATH", None) or None
        vertical  = os.getenv("MARKET_VERTICAL", None)

        if session_id not in _USER_ENVIRONMENTS:
            print(
                f"[APP] New session={session_id} task={task_name} "
                f"vertical={vertical or 'none'}",
                flush=True,
            )
            _USER_ENVIRONMENTS[session_id] = AdPlatformEnvironment(
                task=task_name,
                yaml_path=yaml_path,
                vertical=vertical,
                _market_provider=_MARKET_PROVIDER,
            )

        return _USER_ENVIRONMENTS[session_id]

    except Exception as e:
        import traceback
        print(f"[FACTORY] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        raise

# Pass the factory function
app = create_app(
    get_ad_platform_env, 
    AdPlatformAction, 
    AdPlatformObservation,
    env_name="ad_platform_env"
)

# --- Grade endpoints ---
@app.get("/grade/budget")
def grade_budget():
    env = get_ad_platform_env("default")
    grader_result = compute_score(env._state)
    score = float(max(0.01, min(0.99, grader_result["final_score"])))
    return {"score": score, "reward": score}

@app.get("/grade/auction")
def grade_auction():
    env = get_ad_platform_env("default")
    grader_result = compute_auction_score(env._state)
    score = float(max(0.01, min(0.99, grader_result["final_score"])))
    return {"score": score, "reward": score}

@app.get("/grade/dynamic_campaign")
def grade_dynamic_campaign():
    env = get_ad_platform_env("default")
    grader_result = compute_dynamics_campaign_score(env._state)
    score = float(max(0.01, min(0.99, grader_result["final_score"])))
    return {"score": score, "reward": score}

################################################ Readme.MD display for main web page ####################################################################

@app.get("/", response_class=HTMLResponse)
def root():
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text() if readme_path.exists() else "# Ad Platform RL Environment"
    # Escape backticks for JS template literal
    content_escaped = content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Ad Platform RL Environment</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    <style>
        .markdown-body {{ max-width: 900px; margin: 40px auto; padding: 0 20px; }}
    </style>
</head>
<body class="markdown-body">
    <div id="content"></div>
    <script>
        document.getElementById('content').innerHTML = marked.parse(`{content_escaped}`);
    </script>
</body>
</html>"""

##############################################################################################################################################################

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m ad_platform_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn ad_platform_env.server.app:app --yaml-path /path/to/your/yaml.yaml --host 0.0.0.0 --port 8000
    """

    parser = argparse.ArgumentParser(description="Kube SRE Gym server")
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--host", default=host)
    parser.add_argument("--task", choices=("budget", "auction", "dynamic_campaign"), default=None,
                        help="choose task type")
    parser.add_argument("--yaml-path", default=None,
                        help="yaml path for user/custom ad campaign profile")
    parser.add_argument("--vertical", default=None,
                        help="market vertical for the RL env, can be cahnged for each episode, one of str — ecommerce | saas | travel | finance")
    parser.add_argument("--sampling", choices=("sequential", "random"), default=None,
                    help="market data sampling mode: sequential (default) or random")
    parser.add_argument("--refresh-per-episode", action="store_true", default=False,
                    help="refresh market data window each episode (default: False)")
    
    args = parser.parse_args()
    

    if args.task:
        os.environ["TASK"] = args.task
    if args.yaml_path:
        os.environ["YAML_PATH"] = args.yaml_path
    if args.vertical:
        os.environ["MARKET_VERTICAL"] = args.vertical
    if args.sampling:
        os.environ["MARKET_SAMPLING"] = args.sampling
    if args.refresh_per_episode:
        os.environ["MARKET_REFRESH_PER_EPISODE"] = "true" if args.refresh_per_episode else "false"

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":

    main()

