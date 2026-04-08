# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the RL trading environment."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import RlTradingAction, RlTradingObservation


class RlTradingEnv(EnvClient[RlTradingAction, RlTradingObservation, State]):
    """Typed client for the trading environment server."""

    def _step_payload(self, action: RlTradingAction) -> dict[str, Any]:
        """Convert an action model into the request payload."""
        return {"action": action.action.value}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[RlTradingObservation]:
        """Parse a server response into a typed observation."""
        obs_data = payload.get("observation", {})
        observation = RlTradingObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        """Parse server state into the shared OpenEnv state model."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
