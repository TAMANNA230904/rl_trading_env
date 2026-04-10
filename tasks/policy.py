"""LLM prompting and action selection utilities for trading benchmark tasks."""

from __future__ import annotations

import textwrap
from typing import Optional

from openai import OpenAI

from rl_trading_env import TradingActionType
from .registry import TASK_REGISTRY

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a disciplined single-asset trading agent operating in a constrained simulator.

    Objective:
    Maximize risk-adjusted portfolio growth over the episode.

    Hard rules:
    - Choose exactly one action from the valid action set.
    - Never invent an action outside HOLD, BUY, SELL.
    - Respect momentum, moving averages, RSI, current holdings, and available cash.
    - Prefer HOLD when the signal is weak or contradictory.
    - Avoid overtrading.
    - Output exactly one token and nothing else: HOLD or BUY or SELL.

    Difficulty guidance:
    - easy: follow simple short-term direction and avoid obvious mistakes.
    - medium: use trend confirmation from SMA and RSI before entering or exiting.
    - hard: balance trend, mean reversion risk, and inventory preservation.
    """
).strip()


def build_user_prompt(step: int, difficulty: str, observation) -> str:
    valid_actions = ",".join(action.value for action in observation.valid_actions)
    metadata = observation.metadata or {}
    task_id = str(metadata.get("task_id", ""))
    task_def = TASK_REGISTRY.get(task_id, {})
    price_window = [round(float(price), 4) for price in observation.price_window[-12:]]
    current_price = float(observation.current_price)
    sma_short = "None" if observation.sma_short is None else f"{float(observation.sma_short):.4f}"
    sma_long = "None" if observation.sma_long is None else f"{float(observation.sma_long):.4f}"
    rsi = "None" if observation.rsi is None else f"{float(observation.rsi):.2f}"

    return textwrap.dedent(
        f"""
        Task id: {task_id}
        Task difficulty: {difficulty}
        Scenario name: {task_def.get("name", "Unknown")}
        Scenario objective: {task_def.get("description", "Maximize portfolio growth while respecting constraints.")}
        Step: {step}
        Current price: {current_price:.4f}
        Recent prices: {price_window}
        Balance: {float(observation.balance):.4f}
        Shares held: {int(observation.shares_held)}
        Portfolio value: {float(observation.portfolio_value):.4f}
        SMA short: {sma_short}
        SMA long: {sma_long}
        RSI: {rsi}
        Valid actions: {valid_actions}
        Diagnostics: {metadata}

        Decision policy:
        - Treat the scenario objective as part of the benchmark contract.
        - BUY only when valid and the setup is favorable.
        - SELL only when valid and downside or exit conditions are clearer than upside.
        - Otherwise HOLD.

        Return exactly one token: HOLD, BUY, or SELL.
        """
    ).strip()


def score_from_rewards(rewards: list[float], step_cap: int) -> float:
    if not rewards:
        return 0.0
    positive_reward = sum(max(reward, 0.0) for reward in rewards)
    normalized = positive_reward / max(float(step_cap), 1.0)
    return min(max(normalized, 0.0), 0.999)


def heuristic_action(observation, difficulty: str) -> TradingActionType:
    valid_actions = observation.valid_actions or [TradingActionType.HOLD]
    price_window = [float(price) for price in observation.price_window]
    current_price = float(observation.current_price)
    sma_short = observation.sma_short
    sma_long = observation.sma_long
    rsi = observation.rsi
    has_position = observation.shares_held > 0

    if len(price_window) >= 2:
        recent_change = (price_window[-1] - price_window[-2]) / max(price_window[-2], 1e-8)
    else:
        recent_change = 0.0

    if difficulty == "easy":
        if TradingActionType.BUY in valid_actions and not has_position and recent_change > 0:
            return TradingActionType.BUY
        if TradingActionType.SELL in valid_actions and has_position and recent_change < 0:
            return TradingActionType.SELL
        return TradingActionType.HOLD if TradingActionType.HOLD in valid_actions else valid_actions[0]

    if difficulty == "medium":
        bullish = sma_short is not None and sma_long is not None and sma_short >= sma_long
        bearish = sma_short is not None and sma_long is not None and sma_short < sma_long
        if TradingActionType.BUY in valid_actions and not has_position and bullish and (rsi is None or rsi < 70):
            return TradingActionType.BUY
        if TradingActionType.SELL in valid_actions and has_position and (bearish or (rsi is not None and rsi > 72)):
            return TradingActionType.SELL
        return TradingActionType.HOLD if TradingActionType.HOLD in valid_actions else valid_actions[0]

    if TradingActionType.SELL in valid_actions and has_position and (recent_change < -0.004 or (rsi is not None and rsi > 68)):
        return TradingActionType.SELL
    if (
        TradingActionType.BUY in valid_actions
        and not has_position
        and sma_short is not None
        and sma_long is not None
        and sma_short > sma_long
        and current_price <= sma_short * 1.01
    ):
        return TradingActionType.BUY
    return TradingActionType.HOLD if TradingActionType.HOLD in valid_actions else valid_actions[0]


def parse_action(text: str, observation, difficulty: str) -> TradingActionType:
    valid_actions = observation.valid_actions or [TradingActionType.HOLD]
    normalized = (text or "").strip().upper()
    tokens = [part.strip(".,;:!?()[]{}\"'") for part in normalized.split()]
    for token in tokens:
        if token in TradingActionType._value2member_map_:
            candidate = TradingActionType(token)
            if candidate in valid_actions:
                return candidate
    for candidate in TradingActionType:
        if candidate.value in normalized and candidate in valid_actions:
            return candidate
    return heuristic_action(observation, difficulty)


def choose_action(
    client: Optional[OpenAI],
    model_name: Optional[str],
    temperature: float,
    max_tokens: int,
    step: int,
    difficulty: str,
    observation,
) -> TradingActionType:
    if client is None or not model_name:
        return heuristic_action(observation, difficulty)

    metadata = observation.metadata or {}
    if metadata.get("llm_unavailable"):
        return heuristic_action(observation, difficulty)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(step, difficulty, observation)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (completion.choices[0].message.content or "").strip()
        return parse_action(content, observation, difficulty)
    except Exception as e:
        metadata["llm_unavailable"] = True
        print(f"LLM FAILED: difficulty={difficulty} error={e}", flush=True)
        return heuristic_action(observation, difficulty)
