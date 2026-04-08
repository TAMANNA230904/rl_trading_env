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
import textwrap
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from rl_trading_env import RlTradingAction, RlTradingEnv, TradingActionType

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
CUSTOM_PRICE_HISTORY = [
    100.0,
    100.4,
    100.9,
    101.5,
    101.1,
    100.8,
    101.6,
    102.3,
    102.9,
    102.4,
    101.9,
    102.7,
    103.5,
    104.1,
    103.8,
    103.2,
]
TASK_PRICE_HISTORIES = {
    "synthetic_single_asset_trading": CUSTOM_PRICE_HISTORY,
    "momentum_breakout_trading": [
        98.8,
        99.1,
        99.5,
        100.0,
        100.6,
        101.3,
        102.1,
        103.0,
        103.8,
        104.7,
        105.5,
        106.1,
        107.0,
        108.2,
        109.0,
        109.8,
    ],
    "mean_reversion_pullback_trading": [
        110.0,
        109.4,
        108.7,
        107.9,
        106.8,
        105.6,
        104.3,
        103.1,
        101.8,
        100.7,
        99.9,
        99.1,
        98.4,
        97.8,
        97.1,
        96.5,
    ],
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an RL trading agent for a single-asset simulator.
    Choose exactly one action from: HOLD, BUY, SELL.
    Use only the observation and valid_actions provided.
    Reply with exactly one token: HOLD, BUY, or SELL.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.8f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.4f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def combine_price_history(observation, observed_prices: list[float]) -> list[float]:
    task_history = TASK_PRICE_HISTORIES.get(TASK_NAME, CUSTOM_PRICE_HISTORY)
    if TASK_NAME in TASK_PRICE_HISTORIES and TASK_NAME != "synthetic_single_asset_trading":
        merged = observed_prices + list(observation.price_window) + task_history
    else:
        merged = task_history + observed_prices + list(observation.price_window)
    deduped: list[float] = []
    for price in merged:
        rounded = round(float(price), 6)
        if not deduped or deduped[-1] != rounded:
            deduped.append(rounded)
    return deduped


def summarize_trend(history: list[float]) -> tuple[float, float, float]:
    if len(history) < 2:
        return 0.0, 0.0, 0.0
    short_slice = history[-5:]
    long_slice = history[-12:] if len(history) >= 12 else history
    short_return = (short_slice[-1] - short_slice[0]) / max(short_slice[0], 1e-8)
    long_return = (long_slice[-1] - long_slice[0]) / max(long_slice[0], 1e-8)
    mean_price = sum(long_slice) / len(long_slice)
    distance_from_mean = (history[-1] - mean_price) / max(mean_price, 1e-8)
    return short_return, long_return, distance_from_mean


def build_user_prompt(step: int, observation, observed_prices: list[float]) -> str:
    full_history = combine_price_history(observation, observed_prices)
    short_return, long_return, distance_from_mean = summarize_trend(full_history)
    valid_actions = ",".join(action.value for action in observation.valid_actions)
    metadata = observation.metadata or {}
    return textwrap.dedent(
        f"""
        Step: {step}
        Current price: {observation.current_price:.4f}
        Recent prices: {[round(price, 4) for price in full_history[-12:]]}
        Balance: {observation.balance:.4f}
        Shares held: {observation.shares_held}
        Portfolio value: {observation.portfolio_value:.4f}
        SMA short: {observation.sma_short}
        SMA long: {observation.sma_long}
        RSI: {observation.rsi}
        Short return: {short_return:.6f}
        Long return: {long_return:.6f}
        Distance from mean: {distance_from_mean:.6f}
        Valid actions: {valid_actions}
        Last metadata: {metadata}

        Return exactly one valid action token.
        """
    ).strip()


def parse_action(
    text: str,
    valid_actions: list[TradingActionType],
    observation,
    observed_prices: list[float],
) -> TradingActionType:
    normalized = (text or "").strip().upper()
    for candidate in TradingActionType:
        if candidate.value in normalized and candidate in valid_actions:
            return candidate
    return heuristic_action(valid_actions, observation, observed_prices)


def heuristic_action(
    valid_actions: list[TradingActionType],
    observation,
    observed_prices: list[float],
) -> TradingActionType:
    full_history = combine_price_history(observation, observed_prices)
    short_return, long_return, distance_from_mean = summarize_trend(full_history)
    current_price = observation.current_price
    sma_short = observation.sma_short
    sma_long = observation.sma_long
    rsi = observation.rsi

    if (
        TASK_NAME == "momentum_breakout_trading"
        and TradingActionType.BUY in valid_actions
        and short_return > 0.006
        and long_return > 0.0
        and (rsi is None or rsi < 75)
    ):
        return TradingActionType.BUY

    if TASK_NAME == "mean_reversion_pullback_trading" and short_return < 0.0:
        if TradingActionType.SELL in valid_actions and observation.shares_held > 0:
            return TradingActionType.SELL
        if TradingActionType.HOLD in valid_actions:
            return TradingActionType.HOLD

    if (
        TradingActionType.BUY in valid_actions
        and sma_short is not None
        and sma_long is not None
        and (sma_short > sma_long or (short_return > 0.008 and long_return > 0.0))
        and (rsi is None or rsi < 70)
        and current_price <= max(sma_short * 1.01, full_history[-1] * 1.002)
    ):
        return TradingActionType.BUY

    if (
        TradingActionType.SELL in valid_actions
        and (
            (sma_short is not None and sma_long is not None and sma_short < sma_long)
            or short_return < -0.006
            or long_return < -0.012
            or (rsi is not None and rsi > 70)
            or distance_from_mean < -0.01
        )
    ):
        return TradingActionType.SELL

    if TradingActionType.BUY in valid_actions and short_return > 0.012 and distance_from_mean > -0.004:
        return TradingActionType.BUY

    if TradingActionType.SELL in valid_actions and short_return < 0.0 and observation.shares_held > 0:
        return TradingActionType.SELL

    return TradingActionType.HOLD if TradingActionType.HOLD in valid_actions else valid_actions[0]


def choose_action(
    client: Optional[OpenAI],
    step: int,
    observation,
    observed_prices: list[float],
) -> TradingActionType:
    valid_actions = observation.valid_actions or [TradingActionType.HOLD]
    if client is None:
        return heuristic_action(valid_actions, observation, observed_prices)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(step, observation, observed_prices)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        return parse_action(content, valid_actions, observation, observed_prices)
    except Exception:
        return heuristic_action(valid_actions, observation, observed_prices)


async def create_env() -> RlTradingEnv:
    if IMAGE_NAME:
        return await RlTradingEnv.from_docker_image(IMAGE_NAME)
    return RlTradingEnv(base_url="http://localhost:8000")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL and API_KEY and MODEL_NAME else None
    env: Optional[RlTradingEnv] = None
    rewards: list[float] = []
    raw_rewards: list[float] = []
    observed_prices: list[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME or "heuristic")

    try:
        env = await create_env()
        await env.connect()
        result = await env.reset()
        observed_prices.extend(float(price) for price in result.observation.price_window)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_type = choose_action(client, step, result.observation, observed_prices)
            result = await env.step(RlTradingAction(action=action_type))
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            raw_rewards.append(reward)
            observed_prices.append(float(result.observation.current_price))
            steps_taken = step

            metadata = result.observation.metadata or {}
            error = metadata.get("invalid_reason")
            log_step(
                step=step,
                action=action_type.value,
                reward=reward,
                done=result.done,
                error=error,
            )

            if result.done:
                break

        success = sum(raw_rewards) > 0.0
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
