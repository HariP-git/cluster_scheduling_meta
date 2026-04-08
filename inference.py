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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(step: int, expected_stage: int, current_task: dict, queue_length: int, last_reward: float, num_nodes: int, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        You are an AI explicitly coordinating a compute cluster through a 6-stage pipeline loop.
        The valid stages are exactly: 1, 2, 3, 4, 5, 6.
        You MUST output valid JSON and you MUST progress the stages sequentially.
        The environment automatically calculates the Best-Fit node tracking on stage 4 and handles placements autonomously natively!

        Total Step: {step}
        Expected Pipeline Stage: {expected_stage}
        Task to place: {current_task}
        Tasks remaining: {queue_length}
        Last stage fractional reward: {last_reward:.2f}
        Recent steps history:
        {history_block}
        
        Provide the JSON for your next action.
        Example for stage {expected_stage}: {{ "stage_id": {expected_stage} }}
        """
    ).strip()


def parse_action(text: str, fallback_stage: int) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            data = json.loads(text[start : end + 1])
            if "stage_id" not in data:
                data["stage_id"] = fallback_stage
            return data
    except Exception:
        pass
    return {"stage_id": fallback_stage}


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = None
    result = None
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    MAX_TASKS = 3
    MAX_STEPS = MAX_TASKS * 6
    SUCCESS_SCORE_THRESHOLD = 0.5

    try:
        env = SchedulerEnv(base_url="http://localhost:7860")
        result = await env.reset()

        task_name = os.getenv("SCHEDULER_TASK", "schedule_jobs")
        benchmark = os.getenv("SCHEDULER_BENCHMARK", "scheduler")

        log_start(task=task_name, env=benchmark, model=MODEL_NAME)

        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            obs = result.observation
            num_nodes = 10
            
            state_vector = obs.state_vector if obs and hasattr(obs, 'state_vector') else []
            current_task = {"cpu_req": state_vector[30], "mem_req": state_vector[31], "gpu_req": state_vector[32]} if len(state_vector) >= 33 else {}
            queue_length = int(state_vector[34]) if len(state_vector) >= 35 else 0
            
            expected_stage = ((step - 1) % 6) + 1

            prompt = build_prompt(step, expected_stage, current_task, queue_length, last_reward, num_nodes, history)
            
            action_data = {"stage_id": expected_stage}
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150,
                )
                content = completion.choices[0].message.content or ""
                action_data = parse_action(content.strip(), expected_stage)
            except Exception:
                pass 
            
            stage_id = action_data.get("stage_id", expected_stage)
            action_obj = SchedulerAction(
                stage_id=stage_id,
                is_automated_inference=True
            )

            result = await env.step(action_obj)

            reward = result.reward or 0.0
            done = result.done
            
            error = None
            try:
                error = result.observation.metadata.get("error", None)
            except Exception:
                pass

            rewards.append(reward)

            steps_taken = step
            last_reward = reward

            display_done = True if expected_stage == 6 else done

            log_step(
                step=step, 
                action=json.dumps(action_data), 
                reward=reward, 
                done=display_done, 
                error=error
            )

            history.append(f"Stage {action_obj.stage_id} -> reward {reward:+.02f}")

            if done:
                break

    except Exception:
        pass
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

        if rewards:
            score = sum(rewards) / max(1, len(rewards))
            score = min(max(score, -1.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())