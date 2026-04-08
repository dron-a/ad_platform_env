"""
test_connection.py
==================
Tests WebSocket connection and traces competitor_bids through entire chain.
"""

import asyncio
import json
import os
import re
import types

IMAGE_NAME = os.getenv("IMAGE_NAME")


async def main():
    print(f"Testing with IMAGE_NAME={IMAGE_NAME}")

    if not IMAGE_NAME:
        print("ERROR: IMAGE_NAME not set")
        return

    from client import AdPlatformClient
    from models import AdPlatformAction

    env = await AdPlatformClient.from_docker_image(IMAGE_NAME, env_vars={"TASK": os.getenv("TASK", "auction")})
    print(f"Connected: ws={env._ws is not None} url={env._ws_url}")

    # --- Patch _receive to inspect raw wire data ---
    async def debug_receive(self):
        assert self._ws is not None
        raw = await asyncio.wait_for(
            self._ws.recv(), timeout=self._message_timeout
        )
        # Print raw competitor_bids from wire
        match = re.search(r'"competitor_bids"\s*:\s*(\[[^\]]*\])', raw)
        print(f"[RAW WIRE] competitor_bids = {match.group(1) if match else 'NOT FOUND IN RAW'}")
        print(f"[RAW WIRE] first 500 chars: {raw[:500]}")
        return json.loads(raw)

    env._receive = types.MethodType(debug_receive, env)

    try:
        # --- Reset ---
        print("\n--- RESET ---")
        result = await env.reset()
        obs = getattr(result, "observation", result)
        print(f"After parse competitor_bids: {getattr(obs, 'competitor_bids', 'MISSING')}")

        # --- Step ---
        print("\n--- STEP ---")
        action = AdPlatformAction(
            allocations=[33.0, 22.0, 11.0],
            bids=[0.6, 0.5, 0.4]
        )
        result = await env.step(action)
        obs = getattr(result, "observation", result)
        print(f"After parse competitor_bids: {getattr(obs, 'competitor_bids', 'MISSING')}")
        print(f"reward: {getattr(result, 'reward', 'MISSING')}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())