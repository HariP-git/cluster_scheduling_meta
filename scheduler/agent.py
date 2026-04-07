# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Autonomous Scheduler Agent (DQN Powered).

Advances through the 6-stage scheduling pipeline for each episode.
Uses a Deep Q-Network (PyTorch) to automatically learn optimal Best-Fit assignment strategies.

Usage (requires server running on localhost:7860):
    python -m scheduler.agent

    # Or connect to a remote server:
    python -m scheduler.agent --url http://remote-host:7860

    # Run multiple episodes:
    python -m scheduler.agent --episodes 5
"""

from __future__ import annotations

import argparse
import sys
import random
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .client import SchedulerEnv
from .models import SchedulerAction, SchedulerObservation

# The fixed pipeline stages (must match server)
STAGES = [
    "intake",
    "profiling",
    "matching",
    "assignment",
    "balancing",
    "monitoring",
]

# ─── Deep Q-Network Architecture ─────────────────────────────────────────────

class SchedulerDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(SchedulerDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ─── DQN Agent Wrapper ───────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        self.memory = deque(maxlen=2000)
        
        self.model = SchedulerDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state: torch.Tensor) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.cat([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.cat([x[3] for x in batch])
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        
        # Current Q
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ─── Episode Runner ─────────────────────────────────────────────────────────

def run_episode(
    env: SchedulerEnv,
    episode: int,
    dqn_agent: DQNAgent,
    difficulty: str = "medium",
    verbose: bool = True,
) -> float:
    """Run a single scheduling episode through all pipeline stages with DQN."""
    result = env.reset()
    obs = result.observation

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode {episode} - Task Pipeline")
        print(f"  Pipeline: {' -> '.join(STAGES)}")
        print(f"{'='*60}")

    # Track episode total for display
    total_reward = 0.0
    
    # State observation
    state_tensor = torch.tensor([obs.state_vector], dtype=torch.float32)
    
    # Only assign action variable at Stage 4
    action_idx = 0

    for stage_idx, stage_name in enumerate(STAGES):
        if obs.done and stage_idx != 0:
            break
            
        stage_id = stage_idx + 1
        action_kwargs: dict = {"stage_id": stage_id}

        # Stage 1: pass difficulty so the server generates the right task size
        if stage_id == 1:
            action_kwargs["difficulty"] = difficulty

        # Stage 4: DQN agent picks the node — not the user
        if stage_id == 4:
            action_idx = dqn_agent.act(state_tensor)
            action_kwargs["assign_node_id"] = action_idx

        # Execute step
        result = env.step(SchedulerAction(**action_kwargs))
        next_obs = result.observation
        
        # Use fractional reward for current step display, but model only cares if its assignment.
        step_reward = result.reward if result.reward is not None else 0.0
        
        # If at stage 6, extract and return the aggregated normalized `total_reward` directly from the observation model.
        if stage_id == 6 and getattr(next_obs, 'total_reward', None) is not None:
             total_reward = next_obs.total_reward
        elif stage_id < 6:
             total_reward += step_reward
        
        next_state_tensor = torch.tensor([next_obs.state_vector], dtype=torch.float32)
        done_flag = next_obs.done
        
        # ONLY push to experience replay inside stage 4 (since we focus reward optimization on assignment)
        if stage_id == 4:
             dqn_agent.memory.append((state_tensor, action_idx, step_reward, next_state_tensor, done_flag))
             dqn_agent.train_batch()
        
        obs = next_obs
        state_tensor = next_state_tensor

        if verbose:
            status = "[FAIL]" if step_reward < 0 else ("[PASS]" if step_reward > 0 else "[WARN]")
            if stage_id == 4:
                status += f" node={action_kwargs.get('assign_node_id', '?')}"  
            print(f"  {status} Stage '{stage_name}': reward={step_reward:+.4f}")

    if verbose:
        print(f"\n  Summary:")
        print(f"    Normalized Episode Total : {total_reward:+.4f}")
        print(f"    Pipeline                 : {'COMPLETE' if obs.done else 'INCOMPLETE'}")

    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Autonomous Scheduler Agent (DQN)")
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL of the scheduler server (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final summary",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Task difficulty for stage 1 (default: medium)",
    )
    args = parser.parse_args()

    print(f"Connecting to scheduler at {args.url}...")
    
    # Init PyTorch DQN 
    # State Vector Length (NUM_NODES = 10 -> 30 node params + 3 task + 1 stage + 1 queue = 35)
    # Action Size (NUM_NODES = 10)
    dqn_agent = DQNAgent(state_size=35, action_size=10)

    try:
        with SchedulerEnv(base_url=args.url).sync() as env:
            rewards = []
            for ep in range(1, args.episodes + 1):
                r = run_episode(
                    env, ep, dqn_agent,
                    difficulty=args.difficulty,
                    verbose=not args.quiet,
                )
                rewards.append(r)

            if args.episodes > 1:
                avg = sum(rewards) / len(rewards)
                print(f"\n{'='*60}")
                print(f"  {args.episodes} episodes complete")
                print(f"  Average normalized reward: {avg:+.4f}")
                print(f"  Best normalized reward   : {max(rewards):+.4f}")
                print(f"  Worst normalized reward  : {min(rewards):+.4f}")
                print(f"{'='*60}")

    except ConnectionError:
        print(
            f"\nError: Could not connect to {args.url}\n"
            f"Make sure the server is running:\n"
            f"  cd d:\\meta\\scheduler && uvicorn server.app:app --reload --port 7860",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

