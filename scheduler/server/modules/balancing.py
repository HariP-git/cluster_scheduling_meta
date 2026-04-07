# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 5: Balancing — Analyze load distribution across nodes."""

import math
from dataclasses import dataclass

from .base import ModuleConfig, SchedulerModule


@dataclass
class BalancingConfig(ModuleConfig):
    """Configuration for the balancing stage."""

    imbalance_threshold: float = 0.3


class LoadBalancer(SchedulerModule):
    """
    Computes load balance metrics after assignments.
    Identifies imbalanced nodes where utilization deviates
    significantly from the cluster average.
    """

    def _default_config(self) -> BalancingConfig:
        return BalancingConfig()

    def execute(self, cluster, context: dict) -> dict:
        utilizations = []
        for node in cluster.nodes:
            cpu_u = node.used_cpu / node.total_cpu if node.total_cpu else 0
            mem_u = node.used_memory / node.total_memory if node.total_memory else 0
            gpu_u = node.used_gpu / node.total_gpu if node.total_gpu else 0
            utilizations.append((cpu_u + mem_u + gpu_u) / 3.0)

        avg_util = sum(utilizations) / max(1, len(utilizations))
        variance = sum((u - avg_util) ** 2 for u in utilizations) / max(1, len(utilizations))
        std_dev = math.sqrt(variance)

        # 1.0 = perfectly balanced, 0.0 = maximally imbalanced
        balance_score = max(0.0, 1.0 - std_dev * 2)

        imbalanced = []
        for i, u in enumerate(utilizations):
            if abs(u - avg_util) > self.config.imbalance_threshold:
                imbalanced.append(
                    {"node_id": i, "utilization": round(u, 4), "deviation": round(u - avg_util, 4)}
                )

        self._report = {
            "balance_score": round(balance_score, 4),
            "avg_utilization": round(avg_util, 4),
            "std_deviation": round(std_dev, 4),
            "imbalanced_nodes": len(imbalanced),
        }

        return {
            "balance_score": balance_score,
            "utilization_std_dev": std_dev,
            "imbalanced_nodes": imbalanced,
            "node_utilizations": utilizations,
        }
