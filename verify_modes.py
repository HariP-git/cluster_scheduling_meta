import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scheduler.server.scheduler_environment import SchedulerEnvironment
from scheduler.models import SchedulerAction

def test_manual_mode():
    print("Testing Manual Mode (Swagger style)...")
    env = SchedulerEnvironment()
    env.reset()
    
    # Stage 1: Pass difficulty explicitly (manual mode)
    action = SchedulerAction(stage_id=1, difficulty="easy")
    obs = env.step(action)
    print(f"Step 1 Done: {obs.done}")
    
    # Steps 2-5
    for i in range(2, 6):
        env.step(SchedulerAction(stage_id=i))
        
    # Stage 6
    obs = env.step(SchedulerAction(stage_id=6))
    print(f"Step 6 Done: {obs.done} (Expected: True)")
    assert obs.done == True, "Manual mode should end at stage 6"

def test_inference_mode():
    print("\nTesting Inference Mode (Automated queue style)...")
    env = SchedulerEnvironment()
    env.reset()
    
    # 3 tasks * 6 stages = 18 steps total
    for task_idx in range(3):
        print(f"Task {task_idx + 1}")
        for stage_idx in range(1, 7):
            # No difficulty passed, should use internal queue
            action = SchedulerAction(stage_id=stage_idx)
            obs = env.step(action)
            step_num = task_idx * 6 + stage_idx
            if stage_idx == 6:
                expected_done = (task_idx == 2)
                print(f"  Step {step_num} (Stage 6) Done: {obs.done} (Expected: {expected_done})")
                assert obs.done == expected_done, f"Inference mode done mismatch at step {step_num}"
            else:
                assert obs.done == False

if __name__ == "__main__":
    try:
        test_manual_mode()
        test_inference_mode()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
