# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scheduler Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SchedulerAction, SchedulerObservation


class SchedulerEnv(
    EnvClient[SchedulerAction, SchedulerObservation, State]
):
    """
    Client for the Scheduler Environment.

    Connects to a running scheduler server and provides a typed interface
    for resetting the environment and stepping through pipeline stages.

    Example:
        >>> with SchedulerEnv(base_url="http://localhost:7860") as env:
        ...     result = env.reset()
        ...     print(f"First stage: {result.observation.stage}")
        ...
        ...     result = env.step(SchedulerAction(stage="intake"))
        ...     print(f"Progress: {result.observation.progress:.0%}")

    Example with Docker:
        >>> client = SchedulerEnv.from_docker_image("scheduler-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SchedulerAction(stage="intake"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SchedulerAction) -> Dict:
        """Convert SchedulerAction to JSON payload for step message."""
        return action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else action.dict(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SchedulerObservation]:
        """Parse server response into StepResult[SchedulerObservation]."""
        obs_data = payload.get("observation", {})
        observation = SchedulerObservation(
            state_vector=obs_data.get("state_vector", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            total_reward=obs_data.get("total_reward"),
            info=obs_data.get("info", {}),
            metadata=obs_data.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
