# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 3: Matching — Score candidate nodes for the current task."""

from dataclasses import dataclass

from .base import ModuleConfig, SchedulerModule


@dataclass
class MatchingConfig(ModuleConfig):
    """Configuration for the matching stage."""

    strategy: str = "best_fit"  # "best_fit", "first_fit", "worst_fit"


class NodeMatcher(SchedulerModule):
    """
    Evaluates all nodes against the current task and produces a ranked
    list of candidates based on fit quality (tighter fit = higher score).
    """

    def _default_config(self) -> MatchingConfig:
        return MatchingConfig()

    def execute(self, cluster, context: dict) -> dict:
        classified = context.get("classified_task")
        if not classified:
            self._report = {
                "total_nodes_checked": 0,
                "valid_candidates": 0,
                "tasks_with_no_fit": 1,
            }
            return {"candidates": []}

        candidates = []
        max_free = sum(
            n.available_cpu() + n.available_memory() + n.available_gpu()
            for n in cluster.nodes
        )

        for node in cluster.nodes:
            can_fit = (
                node.available_cpu() >= classified["cpu_req"]
                and node.available_memory() >= classified["mem_req"]
                and node.available_gpu() >= classified["gpu_req"]
            )
            if can_fit:
                remaining = (
                    (node.available_cpu() - classified["cpu_req"])
                    + (node.available_memory() - classified["mem_req"])
                    + (node.available_gpu() - classified["gpu_req"])
                )
                score = 1.0 - (remaining / max(1, max_free))
            else:
                score = -1.0

            candidates.append(
                {"node_id": node.node_id, "can_fit": can_fit, "fit_score": round(score, 4)}
            )

        candidates.sort(key=lambda c: c["fit_score"], reverse=True)
        valid_count = sum(1 for c in candidates if c["can_fit"])

        self._report = {
            "total_nodes_checked": len(candidates),
            "valid_candidates": valid_count,
            "tasks_with_no_fit": 0 if valid_count > 0 else 1,
        }
        return {"candidates": candidates}
