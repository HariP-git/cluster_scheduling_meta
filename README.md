---
title: SchedulerRL
emoji: 🗓️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - rl
---

# Scheduler Environment (OpenEnv)

## Overview
This project implements a **sequential-stage Reinforcement Learning (RL) environment** for cluster scheduling. It simulates a 10-node compute cluster where an agent must intelligently assign incoming tasks to nodes based on CPU, memory, and GPU requirements.

The environment adheres to the **OpenEnv** specification, exposing a FastAPI server that allows agents to interact with the cluster via a standardized 6-stage pipeline.

## Core Features
- **6-Stage Pipeline**: Modular execution flow including Intake, Profiling, Matching, Assignment, Balancing, and Monitoring.
- **Pluggable Modules**: Each stage is handled by a strategy-pattern module, allowing for easy logic overrides.
- **Deep Q-Network (DQN) Support**: Includes a built-in PyTorch-based DQN agent that learns optimal placement policies.
- **Comprehensive API**: Full REST and WebSocket support for local and remote interaction.
- **Hugging Face ready**: Optimized Docker configuration for seamless deployment to Hugging Face Spaces.

## Architecture
The scheduling process follows a rigid sequence of stages:

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

## How to Run

### 1. Local Server
Start the environment server using `uv`:
```bash
uv run server
```
The server will be available at [http://localhost:7860](http://localhost:7860) by default.

### 2. Run Inference
To run the automated inference script (which cycles through easy, medium, and hard tasks):
```bash
uv run python inference.py
```

### 3. Train/Run DQN Agent
To interact with the environment using the built-in DQN agent:
```bash
uv run python -m scheduler.agent --episodes 5
```

### 4. Docker Deployment
Building the container locally:
```bash
docker build -t scheduler-rl .
docker run -p 7860:7860 scheduler-rl
```

## Project Structure
```
.
├── Dockerfile               # Production Docker image
├── README.md                # Project documentation
├── pyproject.toml           # Root project metadata & dependencies
├── inference.py             # Multi-task inference script
└── scheduler/               # Core logic
    ├── agent.py             # DQN RL Agent
    ├── models.py            # Pydantic data models
    └── server/              # FastAPI Server & RL Environment
```
