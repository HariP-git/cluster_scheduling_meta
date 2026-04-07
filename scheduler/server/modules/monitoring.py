# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 6: Monitoring — Compute final cluster health and performance metrics."""

from dataclasses import dataclass

from .base import ModuleConfig, SchedulerModule


@dataclass
class MonitoringConfig(ModuleConfig):
    """Configuration for the monitoring stage."""

    compute_health: bool = True


class ClusterMonitor(SchedulerModule):
    """
    Evaluates the overall quality of the scheduling decisions.
    Produces per-resource utilization, health score, and node summary.
    """

    def _default_config(self) -> MonitoringConfig:
        return MonitoringConfig()

    def execute(self, cluster, context: dict) -> dict:
        successful = context.get("successful_count", 0)
        failed = context.get("failed_count", 0)
        balance_score = context.get("balance_score", 0.0)

        total_cpu_cap = sum(n.total_cpu for n in cluster.nodes)
        total_mem_cap = sum(n.total_memory for n in cluster.nodes)
        total_gpu_cap = sum(n.total_gpu for n in cluster.nodes)

        total_cpu_used = sum(n.used_cpu for n in cluster.nodes)
        total_mem_used = sum(n.used_memory for n in cluster.nodes)
        total_gpu_used = sum(n.used_gpu for n in cluster.nodes)

        cpu_util = total_cpu_used / max(1, total_cpu_cap)
        mem_util = total_mem_used / max(1, total_mem_cap)
        gpu_util = total_gpu_used / max(1, total_gpu_cap)
        overall_util = (cpu_util + mem_util + gpu_util) / 3.0

        success_rate = successful / max(1, successful + failed)
        health = overall_util * 0.4 + balance_score * 0.3 + success_rate * 0.3

        node_summary = []
        for node in cluster.nodes:
            node_summary.append(
                {
                    "node_id": node.node_id,
                    "tasks": len(node.queue),
                    "cpu_used_pct": round(node.used_cpu / node.total_cpu * 100, 1)
                    if node.total_cpu
                    else 0,
                    "mem_used_pct": round(node.used_memory / node.total_memory * 100, 1)
                    if node.total_memory
                    else 0,
                    "gpu_used_pct": round(node.used_gpu / node.total_gpu * 100, 1)
                    if node.total_gpu
                    else 0,
                }
            )

        self._report = {
            "overall_utilization": round(overall_util, 4),
            "cpu_utilization": round(cpu_util, 4),
            "memory_utilization": round(mem_util, 4),
            "gpu_utilization": round(gpu_util, 4),
            "balance_score": round(balance_score, 4),
            "success_rate": round(success_rate, 4),
            "health_score": round(health, 4),
            "node_summary": node_summary,
        }

        return {
            "overall_utilization": overall_util,
            "health_score": health,
            "node_summary": node_summary,
        }
