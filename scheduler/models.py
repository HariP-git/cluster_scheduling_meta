# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Scheduler Environment.

The scheduler environment simulates a cluster of compute nodes with limited
CPU, memory, and GPU resources. An agent advances through a fixed pipeline
of scheduling stages:

    intake → profiling → matching → assignment → balancing → monitoring
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SchedulerAction(Action):
    """Action for the Scheduler environment — advance through pipeline stages.

    For the web UI only ``stage_id`` is shown.  The remaining fields are
    used programmatically by the inference script / DQN agent.
    """

    # ── UI-visible field ─────────────────────────────────────────────────────
    stage_id: int = Field(
        ...,
        description="Pipeline stage (1-6): intake → profiling → matching → assignment → balancing → monitoring",
        ge=1, le=6,
    )

    # ── Agent-internal fields (hidden from UI, still accepted in JSON) ──────
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        None,
        description=(
            "Stage 1 only — controls task size: "
            "easy (cpu/mem/gpu 4), medium (12), hard (24). "
            "Defaults to 'medium' if omitted."
        ),
    )
    assign_node_id: Optional[int] = Field(
        None,
        description="Stage 4 — node index chosen by the DQN agent.",
    )
    task_specs: Optional[Dict[str, int]] = Field(
        None,
        description="Stage 1 only — specific CPU/MEM/GPU override for inference.",
    )

    # Fields that appear in the web UI form
    _UI_FIELDS: ClassVar[tuple[str, ...]] = ("stage_id",)

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return a trimmed schema exposing only ``stage_id`` to the web UI."""
        full = super().model_json_schema(*args, **kwargs)
        props = full.get("properties", {})
        required = full.get("required", [])

        full["properties"] = {k: v for k, v in props.items() if k in cls._UI_FIELDS}
        full["required"] = [r for r in required if r in cls._UI_FIELDS]
        return full


class NodeState(Observation):
    """Resource availability snapshot of a single compute node."""

    node_id: int = Field(default=0, description="Node identifier")
    available_cpu: float = Field(default=0.0, description="Available CPU units")
    available_memory: float = Field(default=0.0, description="Available memory units")
    available_gpu: float = Field(default=0.0, description="Available GPU units")
    tasks_assigned: int = Field(default=0, description="Number of tasks assigned to this node")


class TaskInfo(Observation):
    """Resource requirements of a single task."""

    task_id: int = Field(default=0, description="Task identifier")
    cpu_req: int = Field(default=0, description="CPU units required")
    mem_req: int = Field(default=0, description="Memory units required")
    gpu_req: int = Field(default=0, description="GPU units required")
    duration: int = Field(default=0, description="Task duration in time units")


class SchedulerObservation(Observation):
    """Observation from the Scheduler environment — RL state snapshot."""

    state_vector: List[float] = Field(default_factory=list, description="Flattened continuous state array")
    total_reward: Optional[float] = Field(default=None, description="Normalized total reward (0-1) across all 6 pipeline stages, only present at stage 6")
    info: Dict[str, Any] = Field(default_factory=dict, description="Errors, extra info")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (alias for info)")
