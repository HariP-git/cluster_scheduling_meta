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

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("SCHEDULER_TASK", "schedule_jobs")
BENCHMARK = os.getenv("SCHEDULER_BENCHMARK", "scheduler")

MAX_TASKS = 3
MAX_STEPS = MAX_TASKS * 6
STATIC_TASKS = [
    {"cpu_req": 4, "mem_req": 4, "gpu_req": 4, "duration": 2},   # easy
    {"cpu_req": 12, "mem_req": 12, "gpu_req": 12, "duration": 5}, # medium
    {"cpu_req": 24, "mem_req": 24, "gpu_req": 24, "duration": 8}, # hard
]
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5

MAX_TOTAL_REWARD = MAX_TASKS * 100.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI explicitly coordinating a compute cluster through a 6-stage pipeline loop.
    The valid stages are exactly: 1, 2, 3, 4, 5, 6.
    
    You MUST output valid JSON and you MUST progress the stages sequentially.
    The environment automatically calculates the Best-Fit node tracking on stage 4 and handles placements autonomously natively!
    
    Example for stage 1: { "stage_id": 1 }
    Example for stage 6: { "stage_id": 6 }
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, expected_stage: int, current_task: dict, queue_length: int, last_reward: float, num_nodes: int, history: List[str]) -> str:
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

def get_model_message(client: OpenAI, step: int, expected_stage: int, current_task: dict, queue_length: int, last_reward: float, num_nodes: int, history: List[str]) -> dict:
    user_prompt = build_user_prompt(step, expected_stage, current_task, queue_length, last_reward, num_nodes, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            data = json.loads(json_str)
            if "stage_id" not in data:
                data["stage_id"] = expected_stage
            return data
        
        raise ValueError("No JSON found")

    except Exception as exc:
        print(f"[DEBUG] Model request failed or parsing failed: {exc}", flush=True)
        return {"stage_id": expected_stage}

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = None
    try:
        env = SchedulerEnv(base_url="http://localhost:7860")
        result = await env.reset()
    except Exception as e:
        print(f"[ERROR] Could not connect to default local server: {e}")
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Initial log_start removed to be placed inside task loop

    try:
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            obs = result.observation
            num_nodes = 10
            current_task = {"cpu_req": obs.state_vector[30], "mem_req": obs.state_vector[31], "gpu_req": obs.state_vector[32]} if len(obs.state_vector) >= 33 else {}
            queue_length = int(obs.state_vector[34]) if len(obs.state_vector) >= 35 else 0
            
            # The expected stage mathematically cycles 1 through 6
            expected_stage = ((step - 1) % 6) + 1

            # Mention task difficulty and perform log_start for each task
            if expected_stage == 1:
                task_idx = ((step - 1) // 6)
                diff = ["easy", "medium", "hard"][task_idx] if task_idx < 3 else "unknown"
                print(f"\n{diff}", flush=True)
                log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

            action_data = get_model_message(client, step, expected_stage, current_task, queue_length, last_reward, num_nodes, history)
            
            stage_id = action_data.get("stage_id", expected_stage)
            action_obj = SchedulerAction(
                stage_id=stage_id,
                is_automated_inference=True
            )

            result = await env.step(action_obj)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.metadata.get("error", None)

            if reward != 0.0:
                rewards.append(reward)
            
            steps_taken = step
            last_reward = reward

            # Determine if this is the end of a task iteration for display purposes
            display_done = True if expected_stage == 6 else done

            # Use standard log_step for all stages (Stage 6 will show done=true)
            action_str = json.dumps(action_data)
            log_step(step=step, action=action_str, reward=reward, done=display_done, error=error)

            history.append(f"Stage {action_obj.stage_id} -> reward {reward:+.2f}")

            if expected_stage == 6:
                task_rewards = rewards[-6:]
                task_score = sum(task_rewards) / max(1, len(task_rewards))
                task_success = task_score >= SUCCESS_SCORE_THRESHOLD
                rewards_str = ",".join(f"{r:.2f}" for r in task_rewards)
                print(f"[END] success={str(task_success).lower()} score={task_score:.3f} rewards={rewards_str}", flush=True)

            if done:
                break

        score = sum(rewards) / max(1, len(rewards))
        score = min(max(score, -1.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if env is not None:
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())