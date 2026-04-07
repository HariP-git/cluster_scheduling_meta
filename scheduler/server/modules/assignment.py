# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 4: Assignment — Assign the current task to a node."""

from dataclasses import dataclass

from .base import ModuleConfig, SchedulerModule


@dataclass
class AssignmentConfig(ModuleConfig):
    """Configuration for the assignment stage."""

    strategy: str = "best_fit"
    max_retries: int = 4


class TaskAssigner(SchedulerModule):
    """
    Assigns cluster.current_task to a node.

    Priority order:
      1. Agent-provided node id (from DQN) if it can fit the task.
      2. Best-fit candidate from the ranked list produced by matching.
      3. Retry across all nodes (shuffled) up to max_retries.

    If no node can fit the task, the task is dropped (failure).
    """

    def _default_config(self) -> AssignmentConfig:
        return AssignmentConfig()

    def execute(self, cluster, context: dict) -> dict:
        task = cluster.current_task
        if not task:
            self._report = {"success": False, "reason": "no_task"}
            return {
                "assigned_node_id": None,
                "success": False,
                "successful_count": 0,
                "failed_count": 0,
            }

        candidates = context.get("candidates", [])
        agent_node_id = context.get("assign_node_id")  # DQN-provided override

        assigned = False
        assigned_node_id = None

        # 1️⃣ Try agent's DQN choice first
        if agent_node_id is not None and 0 <= agent_node_id < len(cluster.nodes):
            node = cluster.nodes[agent_node_id]
            if node.can_run(task):
                node.assign_task(task)
                assigned = True
                assigned_node_id = agent_node_id

        # 2️⃣ Fall back to best-fit from matching stage candidates
        if not assigned:
            for candidate in candidates:
                if not candidate["can_fit"]:
                    continue
                node = cluster.nodes[candidate["node_id"]]
                if node.can_run(task):
                    node.assign_task(task)
                    assigned = True
                    assigned_node_id = candidate["node_id"]
                    break

        # 3️⃣ Retry across all nodes (shuffled)
        if not assigned:
            import random
            for _ in range(self.config.max_retries):
                nodes_shuffled = list(cluster.nodes)
                random.shuffle(nodes_shuffled)
                for node in nodes_shuffled:
                    if node.can_run(task):
                        node.assign_task(task)
                        assigned = True
                        assigned_node_id = node.node_id
                        break
                if assigned:
                    break

        self._report = {
            "success": assigned,
            "assigned_node_id": assigned_node_id,
            "task_id": task.task_id,
            "agent_choice": agent_node_id,
        }

        return {
            "assigned_node_id": assigned_node_id,
            "success": assigned,
            "successful_count": 1 if assigned else 0,
            "failed_count": 0 if assigned else 1,
        }
