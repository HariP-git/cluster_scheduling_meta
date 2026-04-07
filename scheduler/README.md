---
title: Scheduler Environment Server
emoji: 🗓️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Scheduler Environment

A **sequential-stage RL environment** that simulates a 10-node compute cluster with limited CPU, memory, and GPU resources. A **Deep Q-Network (DQN) agent** advances through a fixed 6-stage pipeline and receives per-stage rewards so it can learn optimal scheduling policies.

## Architecture

```
reset() → step(intake) → step(profiling) → step(matching)
       → step(assignment) → step(balancing) → step(monitoring) → done
```

| # | Stage | Module | What It Does |
|---|-------|--------|-------------|
| 1 | `intake`     | `IntakeModule`     | Classify task by resource type, assign priority |
| 2 | `profiling`  | `ResourceProfiler` | Analyze cluster utilization and bottlenecks |
| 3 | `matching`   | `NodeMatcher`      | Score and rank candidate nodes for the task |
| 4 | `assignment` | `TaskAssigner`     | Assign task to best-fit node (DQN guides this) |
| 5 | `balancing`  | `LoadBalancer`     | Evaluate load distribution balance |
| 6 | `monitoring` | `ClusterMonitor`   | Compute final health and utilization metrics |

## Quick Start

```python
from scheduler import SchedulerAction, SchedulerEnv

with SchedulerEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()

    for stage_id in range(1, 7):
        result = env.step(SchedulerAction(stage_id=stage_id))
        print(f"  stage {stage_id}: reward={result.reward:+.4f}")

    print(f"Done! total_reward={result.observation.total_reward:.4f}")
```

## Building the Docker Image

```bash
docker build -t scheduler-env:latest -f server/Dockerfile .
```

## Environment Details

### Action

**SchedulerAction** — two fields:
- `stage_id` (int 1–6) — Must match the current pipeline stage (sync validation).
- `assign_node_id` (int, optional) — Stage 4 only: DQN-provided node index.

### Observation

**SchedulerObservation** — flattened RL state:
- `state_vector` (list[float], length 35) — Flattened continuous state:
  - 10 nodes × 3 values (cpu_free%, mem_free%, gpu_free%) = 30 elements
  - 3 task resource requirements (cpu_req, mem_req, gpu_req)
  - 1 current stage index (1–6)
  - 1 queue flag (1 if task present, else 0)
- `total_reward` (float, only at stage 6) — Mean reward across all 6 stages.
- `done` (bool) — True after stage 6 completes.
- `reward` (float) — Per-stage reward, normalized to [0, 1].
- `metadata` (dict) — Stage reports from each completed module.

### Per-Stage Rewards

| Stage | Reward Signal |
|-------|--------------|
| 1 Intake     | Normalized task demand (`demand / 60`) |
| 2 Profiling  | Cluster free-capacity ratio |
| 3 Matching   | Fraction of nodes that can fit the task |
| 4 Assignment | Best-fit efficiency (`1 - wastage`) |
| 5 Balancing  | `1 - utilization_variance × 20` |
| 6 Monitoring | Placement success rate |

`total_reward` = mean of all 6 stage rewards.

### DQN Agent

The bundled agent uses a **PyTorch Deep Q-Network**:
- **State**: 35-element vector from `state_vector`
- **Actions**: node indices 0–9 (passed as `assign_node_id` at stage 4)
- **Architecture**: 3-layer MLP (35 → 128 → 128 → 10)
- **Training**: Experience replay with epsilon-greedy exploration, MSE loss, Adam optimizer

## Running the Agent

```bash
# Terminal 1 — start the server
cd d:\meta\scheduler
uvicorn server.app:app --reload --port 8000

# Terminal 2 — run the DQN agent
uv run python -m scheduler.agent
uv run python -m scheduler.agent --episodes 10
uv run python -m scheduler.agent --url http://remote-host:8000 --quiet
```

## Development & Testing

### Running Tests

```bash
cd d:\meta\scheduler
uv run pytest test_scheduler.py -v
```

### Direct Environment Testing

```bash
uv run python server/scheduler_environment.py
```

## Project Structure

```
scheduler/
├── __init__.py              # Module exports
├── README.md                # This file
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Project metadata and dependencies (includes torch)
├── models.py                # Action and Observation models
├── client.py                # SchedulerEnv HTTP client
├── agent.py                 # Autonomous DQN scheduling agent
├── test_scheduler.py        # Unit tests
└── server/
    ├── __init__.py           # Server module exports
    ├── scheduler_environment.py  # Core environment (pipeline + domain classes)
    ├── app.py                # FastAPI application
    ├── Dockerfile            # Container image
    └── modules/              # Pluggable pipeline stage modules (Strategy pattern)
        ├── __init__.py
        ├── base.py           # SchedulerModule ABC
        ├── intake.py         # Stage 1: Task classification (single-task)
        ├── profiling.py      # Stage 2: Cluster profiling
        ├── matching.py       # Stage 3: Node candidate ranking (single-task)
        ├── assignment.py     # Stage 4: Task assignment with DQN override
        ├── balancing.py      # Stage 5: Load balance scoring
        └── monitoring.py     # Stage 6: Health monitoring
```
