import asyncio
import json

class MockResult:
    def __init__(self, step):
        self.done = (step == 18)
        self.reward = 0.5
        self.observation = type('obs', (), {'state_vector': [0]*40, 'metadata': {}})()

def log_start():
    print(f"[START] task=mock env=mock model=mock", flush=True)

async def mock_main():
    MAX_TASKS = 3
    MAX_STEPS = MAX_TASKS * 6
    SUCCESS_SCORE_THRESHOLD = 0.5
    rewards = []
    
    for step in range(1, MAX_STEPS + 1):
        expected_stage = ((step - 1) % 6) + 1
        
        # Simulating updated inference.py logic
        if expected_stage == 1:
            task_idx = ((step - 1) // 6)
            diff = ["easy", "medium", "hard"][task_idx] if task_idx < 3 else "unknown"
            print(f"\n{diff}", flush=True)
            log_start()

        # Mock env step
        result = MockResult(step)
        reward = result.reward
        done = result.done
        rewards.append(reward)
        
        display_done = True if expected_stage == 6 else done
        action_data = {"stage_id": expected_stage}
        action_str = json.dumps(action_data)
        
        # log_step equivalent
        done_val = str(display_done).lower()
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error=null")

        if expected_stage == 6:
            task_rewards = rewards[-6:]
            task_score = sum(task_rewards) / 6.0
            task_success = task_score >= SUCCESS_SCORE_THRESHOLD
            rewards_str = ",".join(f"{r:.2f}" for r in task_rewards)
            print(f"[END] success={str(task_success).lower()} score={task_score:.3f} rewards={rewards_str}", flush=True)

        if done:
            break
            

if __name__ == "__main__":
    asyncio.run(mock_main())
