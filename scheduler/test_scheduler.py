# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# type: ignore
# pyright: reportMissingImports=false
"""
Unit tests for the Scheduler Environment.

Tests the environment directly (no server needed).

Usage:
    cd d:\\meta\\scheduler
    uv run pytest test_scheduler.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from pydantic import ValidationError

from server.scheduler_environment import Cluster, Node, SchedulerEnvironment, STAGES, Task
from models import SchedulerAction, SchedulerObservation


# ─── Domain classes ──────────────────────────────────────────────────────────


class TestTask:
    def test_task_creation(self):
        t = Task(0, 10, 15, 5, 3)
        assert t.task_id == 0
        assert t.cpu_req == 10
        assert t.mem_req == 15
        assert t.gpu_req == 5
        assert t.duration == 3

    def test_task_to_dict(self):
        t = Task(1, 8, 12, 4, 2)
        d = t.to_dict()
        assert d == {"task_id": 1, "cpu_req": 8, "mem_req": 12, "gpu_req": 4, "duration": 2}


class TestNode:
    def test_initial_utilization(self):
        """Nodes start 70% utilized."""
        n = Node(0, 100, 100, 100)
        assert abs(n.available_cpu() - 30.0) < 1e-5
        assert abs(n.available_memory() - 30.0) < 1e-5
        assert abs(n.available_gpu() - 30.0) < 1e-5

    def test_can_run_fits(self):
        n = Node(0, 100, 100, 100)
        t = Task(0, 10, 10, 10, 1)
        assert n.can_run(t) is True

    def test_can_run_too_big(self):
        n = Node(0, 100, 100, 100)
        t = Task(0, 50, 10, 10, 1)  # needs 50 CPU, only 30 free
        assert n.can_run(t) is False

    def test_assign_task(self):
        n = Node(0, 100, 100, 100)
        t = Task(0, 10, 15, 5, 1)
        n.assign_task(t)
        assert len(n.queue) == 1
        assert abs(n.available_cpu() - 20.0) < 1e-5
        assert abs(n.available_memory() - 15.0) < 1e-5
        assert abs(n.available_gpu() - 25.0) < 1e-5


class TestCluster:
    def test_reset(self):
        c = Cluster()
        assert len(c.nodes) == Cluster.NUM_NODES
        assert c.current_task is None
        assert c.done is False

    def test_generate_single_task(self):
        c = Cluster()
        t = c.generate_single_task()
        assert isinstance(t, Task)
        assert t.task_id == 1002
        assert t.cpu_req == 12


# ─── Pipeline stages ─────────────────────────────────────────────────────────


class TestStages:
    def test_stage_count(self):
        assert len(STAGES) == 6

    def test_stage_order(self):
        assert STAGES == [
            "intake", "profiling", "matching",
            "assignment", "balancing", "monitoring",
        ]


# ─── Module unit tests (single-task API) ─────────────────────────────────────


class TestIntakeModule:
    def test_classifies_single_task(self):
        from server.modules.intake import IntakeModule

        c = Cluster()
        c.current_task = Task(1001, 18, 5, 5, 3)  # cpu-heavy
        module = IntakeModule()
        result = module.execute(c, {})

        assert "classified_task" in result
        ct = result["classified_task"]
        assert ct is not None
        assert ct["task_id"] == 1001
        assert ct["category"] == "cpu_heavy"

        report = module.get_report()
        assert report["total_tasks"] == 1

    def test_no_task_returns_none(self):
        from server.modules.intake import IntakeModule
        c = Cluster()
        c.current_task = None
        module = IntakeModule()
        result = module.execute(c, {})
        assert result["classified_task"] is None


class TestResourceProfiler:
    def test_profiles_cluster(self):
        from server.modules.profiling import ResourceProfiler

        c = Cluster()
        module = ResourceProfiler()
        result = module.execute(c, {})

        assert "node_profiles" in result
        assert len(result["node_profiles"]) == Cluster.NUM_NODES
        assert "cluster_utilization" in result
        # All nodes at 70% utilization
        assert abs(result["cluster_utilization"] - 0.7) < 0.01


class TestNodeMatcher:
    def test_matches_single_task_to_nodes(self):
        from server.modules.intake import IntakeModule
        from server.modules.matching import NodeMatcher

        c = Cluster()
        c.current_task = Task(2001, 10, 10, 10, 2)
        context = IntakeModule().execute(c, {})

        module = NodeMatcher()
        result = module.execute(c, context)

        assert "candidates" in result
        assert len(result["candidates"]) == Cluster.NUM_NODES
        # All nodes have 30 free capacity, task needs 10 — all should fit
        assert all(cand["can_fit"] for cand in result["candidates"])

    def test_no_candidates_when_task_too_large(self):
        from server.modules.intake import IntakeModule
        from server.modules.matching import NodeMatcher

        c = Cluster()
        c.current_task = Task(3001, 50, 50, 50, 1)  # needs 50, only 30 free
        context = IntakeModule().execute(c, {})

        module = NodeMatcher()
        result = module.execute(c, context)
        assert not any(cand["can_fit"] for cand in result["candidates"])


class TestTaskAssigner:
    def test_assigns_single_task(self):
        from server.modules.intake import IntakeModule
        from server.modules.matching import NodeMatcher
        from server.modules.assignment import TaskAssigner

        c = Cluster()
        c.current_task = Task(4001, 10, 10, 10, 2)
        ctx = IntakeModule().execute(c, {})
        ctx.update(NodeMatcher().execute(c, ctx))

        module = TaskAssigner()
        result = module.execute(c, ctx)

        assert result["success"] is True
        assert result["assigned_node_id"] is not None
        assert result["successful_count"] == 1
        assert result["failed_count"] == 0

    def test_fails_when_task_too_large(self):
        from server.modules.intake import IntakeModule
        from server.modules.matching import NodeMatcher
        from server.modules.assignment import TaskAssigner

        c = Cluster()
        c.current_task = Task(5001, 50, 50, 50, 1)
        ctx = IntakeModule().execute(c, {})
        ctx.update(NodeMatcher().execute(c, ctx))

        module = TaskAssigner()
        result = module.execute(c, ctx)

        assert result["success"] is False
        assert result["failed_count"] == 1

    def test_agent_node_override(self):
        """DQN agent can directly specify which node to use."""
        from server.modules.assignment import TaskAssigner

        c = Cluster()
        c.current_task = Task(6001, 5, 5, 5, 1)
        ctx = {"candidates": [], "assign_node_id": 0}

        module = TaskAssigner()
        result = module.execute(c, ctx)

        assert result["success"] is True
        assert result["assigned_node_id"] == 0


class TestLoadBalancer:
    def test_computes_balance(self):
        from server.modules.balancing import LoadBalancer

        c = Cluster()
        module = LoadBalancer()
        result = module.execute(c, {})

        assert "balance_score" in result
        # All nodes at same utilization = perfect balance
        assert result["balance_score"] == 1.0


class TestClusterMonitor:
    def test_computes_health(self):
        from server.modules.monitoring import ClusterMonitor

        c = Cluster()
        module = ClusterMonitor()
        result = module.execute(c, {"successful_count": 1, "failed_count": 0, "balance_score": 1.0})

        assert "health_score" in result
        assert "overall_utilization" in result
        assert 0.0 <= result["health_score"] <= 1.0


# ─── Environment integration ─────────────────────────────────────────────────


class TestSchedulerEnvironment:
    def test_reset_returns_observation(self):
        env = SchedulerEnvironment()
        obs = env.reset()
        assert isinstance(obs, SchedulerObservation)
        # 10 nodes * 3 + 3 task + 1 stage + 1 queue = 35
        assert len(obs.state_vector) == 35
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.total_reward is None

    def test_full_pipeline_completes(self):
        env = SchedulerEnvironment()
        env.reset()
        for _ in range(3):
            for stage_id in range(1, 7):
                obs = env.step(SchedulerAction(stage_id=stage_id))
        assert obs.done is True
        assert obs.total_reward is not None
        assert 0.0 <= obs.total_reward <= 1.0

    def test_stage_sync_validation(self):
        """Sending wrong stage_id should return error in metadata."""
        env = SchedulerEnvironment()
        env.reset()
        # Expecting stage 1, send stage 2
        obs = env.step(SchedulerAction(stage_id=2))
        assert "Sync error" in obs.info.get("error", "")

    def test_full_episode_sets_done_flag(self):
        env = SchedulerEnvironment()
        env.reset()
        for _ in range(3):
            for stage_id in range(1, 7):
                obs = env.step(SchedulerAction(stage_id=stage_id))
        assert obs.done is True

    def test_step_after_done_auto_cycles(self):
        """After stage 6, the environment auto-resets stage index so stage 1 starts a new episode."""
        env = SchedulerEnvironment()
        env.reset()
        for _ in range(3):
            for stage_id in range(1, 7):
                env.step(SchedulerAction(stage_id=stage_id))
        # stage_idx resets to 0, so stage_id=1 is valid again — no sync error
        obs = env.step(SchedulerAction(stage_id=1))
        assert "Sync error" not in obs.info.get("error", "")
        assert obs.reward >= 0.0

    def test_rewards_all_in_range(self):
        env = SchedulerEnvironment()
        env.reset()
        for _ in range(3):
            for stage_id in range(1, 7):
                obs = env.step(SchedulerAction(stage_id=stage_id))
                assert 0.0 <= obs.reward <= 1.0

    def test_state_tracks_steps(self):
        env = SchedulerEnvironment()
        env.reset()
        assert env.state.step_count == 0
        env.step(SchedulerAction(stage_id=1))
        assert env.state.step_count == 1
        env.step(SchedulerAction(stage_id=2))
        assert env.state.step_count == 2

    def test_metadata_contains_stage_reports(self):
        env = SchedulerEnvironment()
        env.reset()
        for stage_id in range(1, 7):
            obs = env.step(SchedulerAction(stage_id=stage_id))
        # After full pipeline, metadata should include every stage's report
        assert "intake" in obs.info
        assert "profiling" in obs.info
        assert "matching" in obs.info
        assert "assignment" in obs.info
        assert "balancing" in obs.info
        assert "monitoring" in obs.info

    def test_assignment_with_agent_node_choice(self):
        """Agent can supply assign_node_id to guide the DQN assignment."""
        env = SchedulerEnvironment()
        env.reset()
        env.step(SchedulerAction(stage_id=1))  # intake
        env.step(SchedulerAction(stage_id=2))  # profiling
        env.step(SchedulerAction(stage_id=3))  # matching
        # Agent picks node 0
        obs = env.step(SchedulerAction(stage_id=4, assign_node_id=0))
        assert obs.reward >= 0.0  # valid assignment scores positively

    def test_invalid_stage_id(self):
        with pytest.raises(ValidationError):
            SchedulerAction(stage_id=7)

    def test_state_vector_length_after_reset(self):
        """State vector must always be exactly 35 elements."""
        env = SchedulerEnvironment()
        for _ in range(3):
            obs = env.reset()
            assert len(obs.state_vector) == 35
