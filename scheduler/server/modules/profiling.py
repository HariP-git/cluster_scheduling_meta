# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 2: Profiling — Analyze cluster state and resource availability."""

from dataclasses import dataclass

from .base import ModuleConfig, SchedulerModule


@dataclass
class ProfilingConfig(ModuleConfig):
    """Configuration for the profiling stage."""

    detailed: bool = True


class ResourceProfiler(SchedulerModule):
    """
    Computes per-node utilization percentages, identifies bottleneck
    nodes and resources, and estimates total cluster headroom.
    """

    def _default_config(self) -> ProfilingConfig:
        return ProfilingConfig()

    def execute(self, cluster, context: dict) -> dict:
        node_profiles = []

        for node in cluster.nodes:
            cpu_u = (1.0 - node.available_cpu() / node.total_cpu) if node.total_cpu else 1.0
            mem_u = (1.0 - node.available_memory() / node.total_memory) if node.total_memory else 1.0
            gpu_u = (1.0 - node.available_gpu() / node.total_gpu) if node.total_gpu else 1.0
            avg_u = (cpu_u + mem_u + gpu_u) / 3.0

            node_profiles.append(
                {
                    "node_id": node.node_id,
                    "cpu_utilization": round(cpu_u, 4),
                    "memory_utilization": round(mem_u, 4),
                    "gpu_utilization": round(gpu_u, 4),
                    "avg_utilization": round(avg_u, 4),
                    "available_cpu": round(node.available_cpu(), 2),
                    "available_memory": round(node.available_memory(), 2),
                    "available_gpu": round(node.available_gpu(), 2),
                    "tasks_assigned": len(node.queue),
                }
            )

        cluster_util = sum(p["avg_utilization"] for p in node_profiles) / max(
            1, len(node_profiles)
        )
        bottleneck = max(node_profiles, key=lambda p: p["avg_utilization"])
        least_loaded = min(node_profiles, key=lambda p: p["avg_utilization"])

        headroom = {
            "cpu": sum(p["available_cpu"] for p in node_profiles),
            "memory": sum(p["available_memory"] for p in node_profiles),
            "gpu": sum(p["available_gpu"] for p in node_profiles),
        }

        self._report = {
            "cluster_utilization": round(cluster_util, 4),
            "bottleneck_node": bottleneck["node_id"],
            "least_loaded_node": least_loaded["node_id"],
            "total_headroom": headroom,
        }

        return {
            "node_profiles": node_profiles,
            "cluster_utilization": cluster_util,
            "bottleneck_node_id": bottleneck["node_id"],
            "total_headroom": headroom,
        }
