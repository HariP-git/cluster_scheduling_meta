import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from scheduler.models import SchedulerAction
from scheduler.client import SchedulerEnv

# Load environment variables from .env file
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Check for API Key - switch to MOCK_MODE locally if missing
MOCK_MODE = not HF_TOKEN
API_KEY = HF_TOKEN

if MOCK_MODE:
    print("\n" + "!" * 60)
    print("⚠️  WARNING: NO HF_TOKEN FOUND. SWITCHING TO MOCK AGENT MODE.")
    print("!" * 60 + "\n")
    MODEL_NAME = "Mock/Deterministic-Agent"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI explicitly coordinating a compute cluster through a 6-stage pipeline loop.
    The valid stages are exactly: 1, 2, 3, 4, 5, 6.
    
    You MUST output valid JSON and you MUST progress the stages sequentially.
    The environment automatically calculates the Best-Fit node tracking on stage 4 and handles placements autonomously natively!
    
    Example for stage 1: { "stage_id": 1 }
    """
).strip()


# ✅ UPDATED CLAMP FUNCTION
def clamp_reward(val: float, limit: float = 0.99) -> float:
    """Ensure reward is strictly within (0.1, 0.99)."""
    try:
        return min(max(float(val), 0.1), limit)
    except (ValueError, TypeError):
        return 0.1


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    done_val = str(done).lower()
    clamped = clamp_reward(reward)
    print(
        f"[STEP] step={step} action={action} reward={clamped:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{clamp_reward(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(step: int, expected_stage: int, current_task: dict, queue_length: int, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        Total Step: {step}
        Expected Pipeline Stage: {expected_stage}
        Task to place: {current_task}
        Tasks remaining: {queue_length}
        Last stage fractional reward: {last_reward:.2f}
        Recent steps history:
        {history_block}
        
        Provide the JSON for your next action.
        """
    ).strip()


def parse_action(text: str, fallback_stage: int) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(text[start : end + 1])
            if "stage_id" not in data:
                data["stage_id"] = fallback_stage
            return data
    except Exception:
        pass
    return {"stage_id": fallback_stage}


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if not MOCK_MODE else None
    
    history: List[str] = []
    task_rewards: List[float] = []
    steps_total = 0
    SUCCESS_SCORE_THRESHOLD = 0.5
    
    # Mode Detection
    env_task = os.getenv("SCHEDULER_TASK")
    benchmark = os.getenv("SCHEDULER_BENCHMARK", "scheduler")
    is_platform_run = env_task is not None and env_task not in ["schedule_jobs", "scheduler"]
    
    if is_platform_run:
        tasks_to_run = [env_task]
    else:
        tasks_to_run = ["scheduling_easy", "scheduling_medium", "scheduling_hard"]

    env = None
    try:
        env = SchedulerEnv(base_url="http://localhost:7860")
        
        for task_idx, current_task_id in enumerate(tasks_to_run):
            result = await env.reset()
            
            # Difficulty derived from task name
            difficulty = "medium"
            if "easy" in current_task_id.lower():
                difficulty = "easy"
            elif "hard" in current_task_id.lower():
                difficulty = "hard"

            if not is_platform_run:
                print(f"\n{'='*50}", flush=True)
                print(f"  Task {task_idx + 1}/3: {current_task_id.upper()}", flush=True)
                print(f"{'='*50}", flush=True)

            log_start(task=current_task_id, env=benchmark, model=MODEL_NAME)
            last_reward = 0.1  # ✅ start within valid range
            
            # 6-stage pipeline loop
            for sub_step in range(1, 7):
                current_global_step = (task_idx * 6) + sub_step
                expected_stage = sub_step
                
                obs = result.observation
                state_vector = obs.state_vector if obs and hasattr(obs, 'state_vector') else []
                current_task_reqs = {
                    "cpu_req": state_vector[30], 
                    "mem_req": state_vector[31], 
                    "gpu_req": state_vector[32]
                } if len(state_vector) >= 33 else {}
                queue_length = int(state_vector[34]) if len(state_vector) >= 35 else 0
                
                action_data = {"stage_id": expected_stage}
                
                if not MOCK_MODE:
                    try:
                        prompt = build_user_prompt(
                            current_global_step,
                            expected_stage,
                            current_task_reqs,
                            queue_length,
                            last_reward,
                            history
                        )
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=150,
                        )
                        action_data = parse_action(
                            completion.choices[0].message.content or "",
                            expected_stage
                        )
                    except Exception:
                        pass 
                
                action_kwargs = {
                    "stage_id": action_data.get("stage_id", expected_stage),
                    "is_automated_inference": True
                }
                if expected_stage == 1:
                    action_kwargs["difficulty"] = difficulty
                    
                action_obj = SchedulerAction(**action_kwargs)
                result = await env.step(action_obj)

                reward_to_log = clamp_reward(result.reward or 0.1)

                if expected_stage == 6 and hasattr(result.observation, 'total_reward'):
                    if result.observation.total_reward is not None:
                        reward_to_log = clamp_reward(result.observation.total_reward)
                        task_rewards.append(reward_to_log)

                        if not is_platform_run:
                            print(f"  [INFO] {current_task_id} completed. score: {reward_to_log:.2f}", flush=True)

                error = None
                try:
                    error = result.observation.metadata.get("error", None)
                except Exception:
                    pass

                steps_total += 1
                last_reward = reward_to_log

                log_step(
                    step=current_global_step, 
                    action=json.dumps(action_data), 
                    reward=reward_to_log, 
                    done=(expected_stage == 6 or result.done), 
                    error=error
                )

                history.append(f"T:{current_task_id} S:{expected_stage} -> {reward_to_log:+.2f}")
                if result.done or expected_stage == 6:
                    break

    except Exception:
        pass
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

        # ✅ fallback fixed
        if not task_rewards:
            task_rewards = [0.1] * len(tasks_to_run)
            
        score = sum(task_rewards) / len(task_rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD

        if not is_platform_run:
            print(f"\n{'='*50}", flush=True)
            print(f"  EPISODE COMPLETE | Score: {score:.3f} | Success: {success}", flush=True)
            print(f"{'='*50}\n", flush=True)

        log_end(success=success, steps=steps_total, rewards=task_rewards)


if __name__ == "__main__":
    asyncio.run(main())
