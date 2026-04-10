"""
Inference runner for the RL trading OpenEnv environment.

Environment variables are intentionally kept unchanged:
- IMAGE_NAME: optional local Docker image for the environment
- API_BASE_URL: LLM endpoint base URL
- MODEL_NAME: LLM model name
- HF_TOKEN / API_KEY: LLM API credential
- TASK_NAME / BENCHMARK: metadata used in stdout logs
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from rl_trading_env import RlTradingAction, RlTradingEnv, TradingActionType
from tasks import TASK_CONFIGS, choose_action, score_from_rewards

load_dotenv()

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
TASK_NAME = os.getenv("TASK_NAME", "rl_trading_env")
BENCHMARK = os.getenv("BENCHMARK", "synthetic_single_asset_trading")
MAX_STEPS = int(os.getenv("MAX_STEPS", "64"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16"))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_end(success: bool, steps: int, score: float, total_reward: float) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} total_reward={total_reward:.8f} score={score:.8f}",
        flush=True,
    )
    print(flush=True)


async def create_env() -> RlTradingEnv:
    if IMAGE_NAME:
        return await RlTradingEnv.from_docker_image(IMAGE_NAME)
    return RlTradingEnv(base_url="http://localhost:7860")


async def run_task(client: Optional[OpenAI], task_config: dict[str, object]) -> float:
    env: Optional[RlTradingEnv] = None
    rewards: list[float] = []
    steps_taken = 0
    difficulty = str(task_config["difficulty"])
    task_id = str(task_config["task_id"])
    step_cap = min(MAX_STEPS, int(task_config["target_steps"]))
    score = 0.0
    total_reward = 0.0
    success = False

    log_start(task=task_id, env=f"{BENCHMARK}:{difficulty}", model=MODEL_NAME or "heuristic")

    try:
        env = await create_env()
        await env.connect()
        result = await env.reset(task_id=task_id)

        for step in range(1, step_cap + 1):
            if result.done:
                break

            action_type = choose_action(
                client=client,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                step=step,
                difficulty=difficulty,
                observation=result.observation,
            )
            try:
                result = await env.step(RlTradingAction(action=action_type))
            except Exception as e:
                print(f"ENV STEP FAILED: task={task_id} step={step} action={action_type.value} error={e}", flush=True)
                break

            reward = min(max(float(result.reward or 0.0), 0.0), 0.999)
            rewards.append(reward)
            total_reward += reward
            steps_taken = step

            metadata = result.observation.metadata or {}
            error = metadata.get("invalid_reason")
            if error:
                print(
                    f"[WARN] task={task_id} step={step} action={action_type.value} error={error}",
                    flush=True,
                )

            if result.done:
                break

        score = score_from_rewards(rewards, step_cap)
        success = score > 0.0
    except Exception as e:
        print(f"TASK FAILED: task={task_id} error={e}", flush=True)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                error_message = str(e)
                if "docker" not in error_message.lower() or "timed out" not in error_message.lower():
                    print(f"ENV CLOSE FAILED: task={task_id} error={e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, total_reward=total_reward)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL and API_KEY and MODEL_NAME else None
    print("LLM STATUS:", flush=True)
    print("API_BASE_URL:", API_BASE_URL, flush=True)
    print("API_KEY:", "SET" if API_KEY else "NOT SET", flush=True)
    print("MODEL_NAME:", MODEL_NAME, flush=True)
    print("Client created:", client is not None, flush=True)

    scores: list[float] = []
    for task_config in TASK_CONFIGS:
        score = await run_task(client, task_config)
        scores.append(score)

    total_score = min(max(sum(scores) / max(len(scores), 1), 0.0), 0.999)
    print(f"[SUMMARY] benchmark={BENCHMARK} average_score={total_score:.8f}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FATAL INFERENCE ERROR: {e}", flush=True)
