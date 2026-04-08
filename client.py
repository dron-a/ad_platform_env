from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import AdPlatformAction, AdPlatformObservation


class AdPlatformClient(EnvClient[AdPlatformAction, AdPlatformObservation, State]):
    """Client for the AdPlatform environment."""

    def _step_payload(self, action: AdPlatformAction) -> Dict:
        return action.model_dump()

    # def _parse_result(self, payload: Dict) -> StepResult[AdPlatformObservation]:
    #     print(f"[DEBUG] Raw payload keys: {list(payload.keys())}", flush=True)
        # print(f"[DEBUG] Raw payload competitor_bids: {payload.get('observation', {})}", flush=True)
    #     obs_data = payload.get("observation", {})
    #     obs = AdPlatformObservation.model_validate(obs_data)
    #     print(f"[DEBUG] obs_data keys: {list(obs_data.keys())}", flush=True)
    #     print(f"[DEBUG] competitor_bids in obs_data: {obs_data.get('competitor_bids')}", flush=True)
    #     return StepResult(
    #         observation=obs,
    #         reward=payload.get("reward"),
    #         done=payload.get("done", False),
    #     )

    def _parse_result(self, payload: Dict) -> StepResult[AdPlatformObservation]:
        obs_data = payload.get("observation", {})
        obs = AdPlatformObservation.model_validate(obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )