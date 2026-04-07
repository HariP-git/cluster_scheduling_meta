# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# type: ignore
# pyright: reportMissingImports=false
"""
Scheduler Environment Implementation.

Integrating the 6-stage pipeline loop with pluggable stage modules,
automated best-fit machine tracking and fractional rewards.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SchedulerAction, SchedulerObservation
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models import SchedulerAction, SchedulerObservation

from .modules import (
    IntakeModule,
    ResourceProfiler,
    NodeMatcher,
    TaskAssigner,
    LoadBalancer,
    ClusterMonitor,
)

STAGES = ["intake", "profiling", "matching", "assignment", "balancing", "monitoring"]


# ─── Domain Classes ───────────────────────────────────────────────────────────


class Task:
    def __init__(self, task_id: int, cpu_req: float, mem_req: float, gpu_req: float, duration: int):
        self.task_id = task_id
        self.cpu_req = cpu_req
        self.mem_req = mem_req
        self.gpu_req = gpu_req
        self.duration = duration

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "cpu_req": self.cpu_req,
            "mem_req": self.mem_req,
            "gpu_req": self.gpu_req,
            "duration": self.duration,
        }


class Node:
    def __init__(self, node_id: int, total_cpu: float, total_memory: float, total_gpu: float):
        self.node_id = node_id
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        self.total_gpu = total_gpu
        self.used_cpu = 0.7 * total_cpu
        self.used_memory = 0.7 * total_memory
        self.used_gpu = 0.7 * total_gpu
        self.queue: list[Task] = []

    def available_cpu(self) -> float:
        return self.total_cpu - self.used_cpu

    def available_memory(self) -> float:
        return self.total_memory - self.used_memory

    def available_gpu(self) -> float:
        return self.total_gpu - self.used_gpu

    def can_run(self, task: Task) -> bool:
        return (
            self.available_cpu() >= task.cpu_req
            and self.available_memory() >= task.mem_req
            and self.available_gpu() >= task.gpu_req
        )

    def assign_task(self, task: Task) -> None:
        self.queue.append(task)
        self.used_cpu += task.cpu_req
        self.used_memory += task.mem_req
        self.used_gpu += task.gpu_req


class Cluster:
    NUM_NODES = 10
    NODE_CAPACITY = 100

    def __init__(self):
        self.nodes: list[Node] = []
        self.current_task: Task | None = None
        self.done = False
        self.reset()

    def generate_single_task(self):
        return self.generate_task_by_difficulty("medium")

    def generate_task_by_difficulty(self, difficulty: str = "medium") -> "Task":
        """
        Generate a static task scaled to the requested difficulty:
          easy   — cpu/mem/gpu 4
          medium — cpu/mem/gpu 12
          hard   — cpu/mem/gpu 24
        """
        if difficulty == "easy":
            return Task(1001, 4, 4, 4, 2)
        elif difficulty == "hard":
            return Task(1003, 24, 24, 24, 8)
        else:
            return Task(1002, 12, 12, 12, 5)

    def reset(self):
        self.nodes = [
            Node(i, self.NODE_CAPACITY, self.NODE_CAPACITY, self.NODE_CAPACITY)
            for i in range(self.NUM_NODES)
        ]
        self.current_task = None
        self.done = False


# ─── Environment ──────────────────────────────────────────────────────────────


class SchedulerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.cluster = Cluster()
        self.current_stage_idx = 0
        self.total_evaluated = 0
        self.total_successful = 0
        self.current_fit_quality = 0.0
        self.current_job_success = False
        self.stage_rewards: list[float] = []
        self._pipeline_context: dict = {}
        self._stage_reports: dict = {}

        # Pluggable stage modules (Strategy pattern)
        self._intake = IntakeModule()
        self._profiler = ResourceProfiler()
        self._matcher = NodeMatcher()
        self._assigner = TaskAssigner()
        self._balancer = LoadBalancer()
        self._monitor = ClusterMonitor()
        self.pending_tasks: list[str] = ["easy", "medium", "hard"]
        self.episode_rewards: list[float] = []

    def reset(self) -> SchedulerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.cluster.reset()
        self.current_stage_idx = 0
        self.total_evaluated = 0
        self.total_successful = 0
        self.current_fit_quality = 0.0
        self.current_job_success = False
        self.stage_rewards = []
        self._pipeline_context = {}
        self._stage_reports = {}
        self.pending_tasks: list[str] = ["easy", "medium", "hard"]
        self.episode_rewards: list[float] = []
        return self._build_observation(reward=0.0)

    def step(self, action: SchedulerAction) -> SchedulerObservation:
        self._state.step_count += 1

        if self.cluster.done:
            return self._build_observation(reward=0.0)

        expected = self.current_stage_idx + 1
        if action.stage_id != expected:
            return self._build_observation(
                reward=0.0,
                error=f"Sync error: expected stage {expected}, got {action.stage_id}."
            )

        reward = 0.0
        error_msg = None
        executed_stage = STAGES[action.stage_id - 1]

        # Max possible task resource demand (cpu+mem+gpu each up to 20 → max 60)
        MAX_TASK_DEMAND = 60.0
        # Max per-node total resources (cpu+mem+gpu = 300)
        NODE_TOTAL = self.cluster.NODE_CAPACITY * 3

        # ── Stage execution via pluggable modules ──────────────────────────
        if action.stage_id == 1:  # Intake
            if action.difficulty is not None or action.task_specs is not None:
                self._pipeline_context["manual_mode"] = True
                difficulty = action.difficulty or "medium"
            else:
                self._pipeline_context["manual_mode"] = False
                difficulty = self.pending_tasks.pop(0) if self.pending_tasks else "medium"
                
            difficulty = difficulty.lower()
            
            if action.task_specs:
                self.cluster.current_task = Task(
                    task_id=2000,
                    cpu_req=action.task_specs.get("cpu_req", 5),
                    mem_req=action.task_specs.get("mem_req", 5),
                    gpu_req=action.task_specs.get("gpu_req", 5),
                    duration=action.task_specs.get("duration", 5)
                )
            else:
                self.cluster.current_task = self.cluster.generate_task_by_difficulty(difficulty)
            self.current_fit_quality = 0.0
            self.current_job_success = False
            self._pipeline_context.update({"difficulty": difficulty})

            ctx = self._intake.execute(self.cluster, {"difficulty": difficulty})
            self._pipeline_context.update(ctx)
            self._stage_reports["intake"] = {**self._intake.get_report(), "difficulty": difficulty}

            task = self.cluster.current_task
            demand = task.cpu_req + task.mem_req + task.gpu_req
            reward = round(min(1.0, demand / MAX_TASK_DEMAND), 4)

        elif action.stage_id == 2:  # Profiling
            ctx = self._profiler.execute(self.cluster, self._pipeline_context)
            self._pipeline_context.update(ctx)
            self._stage_reports["profiling"] = self._profiler.get_report()

            total_free = sum(
                n.available_cpu() + n.available_memory() + n.available_gpu()
                for n in self.cluster.nodes
            )
            total_cap = len(self.cluster.nodes) * NODE_TOTAL
            reward = round(total_free / max(1.0, total_cap), 4)

        elif action.stage_id == 3:  # Matching
            ctx = self._matcher.execute(self.cluster, self._pipeline_context)
            self._pipeline_context.update(ctx)
            self._stage_reports["matching"] = self._matcher.get_report()

            eligible = sum(
                1 for c in ctx.get("candidates", []) if c["can_fit"]
            )
            reward = round(eligible / max(1, len(self.cluster.nodes)), 4)

        elif action.stage_id == 4:  # Assignment
            task = self.cluster.current_task
            if task:
                self.total_evaluated += 1

            # Pass agent's DQN node choice into context
            assign_ctx = dict(self._pipeline_context)
            assign_ctx["assign_node_id"] = action.assign_node_id

            ctx = self._assigner.execute(self.cluster, assign_ctx)
            self._pipeline_context.update(ctx)
            self._stage_reports["assignment"] = self._assigner.get_report()

            if ctx.get("success") and task:
                self.total_successful += 1
                self.current_job_success = True
                node_id = ctx["assigned_node_id"]
                node = self.cluster.nodes[node_id]
                rem = node.available_cpu() + node.available_memory() + node.available_gpu()
                wastage = rem / NODE_TOTAL
                # Tight fit = high reward; loose fit = lower but still positive
                reward = round(1.0 - wastage, 4)
                self.current_fit_quality = reward
            else:
                self.current_job_success = False
                # Penalise the agent — negative reward so DQN clearly learns failure
                reward = -0.3
                error_msg = "Task could not be placed after 4 retries."

        elif action.stage_id == 5:  # Balancing
            ctx = self._balancer.execute(self.cluster, self._pipeline_context)
            self._pipeline_context.update(ctx)
            self._stage_reports["balancing"] = self._balancer.get_report()

            utils = [
                (n.used_cpu + n.used_memory + n.used_gpu) / NODE_TOTAL
                for n in self.cluster.nodes
            ]
            avg = sum(utils) / len(utils)
            variance = sum((u - avg) ** 2 for u in utils) / max(1, len(utils))
            reward = round(max(0.0, 1.0 - variance * 20), 4)

        elif action.stage_id == 6:  # Monitoring
            ctx = self._monitor.execute(self.cluster, self._pipeline_context)
            self._pipeline_context.update(ctx)
            self._stage_reports["monitoring"] = self._monitor.get_report()

            reward = round(
                self.total_successful / max(1, self.total_evaluated), 4
            )

        # Stage 4 (assignment) may return a negative penalty; all others clamped to [0, 1]
        if action.stage_id == 4:
            reward = round(max(-1.0, min(1.0, float(reward))), 4)
        else:
            reward = round(max(0.0, min(1.0, float(reward))), 4)
        self.stage_rewards.append(reward)

        # ── Advance or complete ────────────────────────────────────────────
        if action.stage_id == 6:
            task_reward = round(
                sum(self.stage_rewards) / max(1, len(self.stage_rewards)), 4
            )
            self.episode_rewards.append(task_reward)
            
            if self._pipeline_context.get("manual_mode", False):
                is_done = True
            else:
                is_done = len(self.pending_tasks) == 0
            
            # Always provide total_reward for the task at stage 6 so the UI can display it!
            total_reward = task_reward
            
            obs = self._build_observation(
                reward=reward,
                error=error_msg,
                executed_stage=executed_stage,
                total_reward=total_reward,
            )
            obs.done = is_done
            self.current_stage_idx = 0
            self.cluster.current_task = None
            self.stage_rewards = []
            
            if is_done:
                self.total_evaluated = 0
                self.total_successful = 0
                self.pending_tasks = ["easy", "medium", "hard"]
                self.episode_rewards = []
                
            return obs
        else:
            self.current_stage_idx += 1

        return self._build_observation(reward=reward, error=error_msg, executed_stage=executed_stage)

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(
        self,
        reward: float = 0.0,
        error: str | None = None,
        executed_stage: str | None = None,
        total_reward: float | None = None,
    ) -> SchedulerObservation:
        state_vector = []
        for n in self.cluster.nodes:
            state_vector.extend([
                round(n.available_cpu() / max(1, n.total_cpu), 2),
                round(n.available_memory() / max(1, n.total_memory), 2),
                round(n.available_gpu() / max(1, n.total_gpu), 2),
            ])

        task = self.cluster.current_task
        if task:
            state_vector.extend([
                round(task.cpu_req, 1),
                round(task.mem_req, 1),
                round(task.gpu_req, 1),
            ])
        else:
            state_vector.extend([0.0, 0.0, 0.0])

        stage_idx = self.current_stage_idx if self.current_stage_idx < len(STAGES) else len(STAGES) - 1
        state_vector.append(float(stage_idx + 1))

        queue_len = len(self.pending_tasks)
        state_vector.append(float(queue_len))

        md = dict(self._stage_reports)  # include all accumulated stage reports
        
        # Add a human-readable view of the nodes requested by the user
        readable_state = {}
        for n in self.cluster.nodes:
            readable_state[f"Node {n.node_id}"] = [
                round(n.available_cpu() / max(1, n.total_cpu), 2),
                round(n.available_memory() / max(1, n.total_memory), 2),
                round(n.available_gpu() / max(1, n.total_gpu), 2),
            ]
        md["readable_cluster_state_free_pct"] = readable_state
        
        if error:
            md["error"] = error

        return SchedulerObservation(
            state_vector=state_vector,
            done=self.cluster.done,
            reward=reward,
            total_reward=total_reward,
            info=md,
        )


if __name__ == "__main__":
    env = SchedulerEnvironment()
    obs = env.reset()
    print(f"Reset. State vector length: {len(obs.state_vector)}")
    print(f"Expected: {Cluster.NUM_NODES * 3 + 5} elements")
