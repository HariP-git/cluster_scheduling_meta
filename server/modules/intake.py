# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1: Intake — Receive and classify the current incoming task."""

from dataclasses import dataclass

from .base import ModuleConfig, SchedulerModule


@dataclass
class IntakeConfig(ModuleConfig):
    """Configuration for the intake stage."""

    classify_tasks: bool = True
    assign_priorities: bool = True


class IntakeModule(SchedulerModule):
    """
    Classifies the current task by its dominant resource requirement
    (cpu_heavy, memory_heavy, gpu_heavy, balanced) and assigns
    a priority score based on total resource demand and urgency.
    """

    def _default_config(self) -> IntakeConfig:
        return IntakeConfig()

    def execute(self, cluster, context: dict) -> dict:
        task = cluster.current_task
        if not task:
            self._report = {"total_tasks": 0}
            return {"classified_task": None}

        category = self._classify(task)
        priority = self._compute_priority(task)

        classified = {
            "task_id": task.task_id,
            "category": category,
            "priority": priority,
            "cpu_req": task.cpu_req,
            "mem_req": task.mem_req,
            "gpu_req": task.gpu_req,
            "duration": task.duration,
        }

        self._report = {
            "total_tasks": 1,
            "category": category,
            "priority": priority,
            "avg_priority": priority,
        }
        return {"classified_task": classified}

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _classify(task) -> str:
        reqs = {
            "cpu_heavy": task.cpu_req,
            "memory_heavy": task.mem_req,
            "gpu_heavy": task.gpu_req,
        }
        if max(reqs.values()) - min(reqs.values()) <= 3:
            return "balanced"
        return max(reqs, key=reqs.get)

    @staticmethod
    def _compute_priority(task) -> float:
        total_demand = task.cpu_req + task.mem_req + task.gpu_req
        urgency = 1.0 / max(1, task.duration)
        return round(total_demand * 0.7 + urgency * 30 * 0.3, 2)
