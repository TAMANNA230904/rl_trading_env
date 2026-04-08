"""
Deterministic agent task graders for the RL trading project.

This file is separate from `inference.py` so the mentor-provided rollout runner
can stay unchanged while the hackathon task/grader requirement is still met.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

import inference
from rl_trading_env import RlTradingObservation, TradingActionType


@dataclass(frozen=True)
class AgentTaskStep:
    observation: RlTradingObservation
    expected_action: TradingActionType


@dataclass(frozen=True)
class AgentTask:
    name: str
    difficulty: str
    objective: str
    steps: tuple[AgentTaskStep, ...]


def make_observation(
    *,
    current_price: float,
    price_window: list[float],
    balance: float,
    shares_held: int,
    portfolio_value: float,
    valid_actions: list[TradingActionType],
    sma_short: float,
    sma_long: float,
    rsi: float,
    task_objective: str,
    step_index: int,
    difficulty: str,
) -> RlTradingObservation:
    return RlTradingObservation(
        current_price=current_price,
        price_window=price_window,
        balance=balance,
        shares_held=shares_held,
        portfolio_value=portfolio_value,
        sma_short=sma_short,
        sma_long=sma_long,
        rsi=rsi,
        valid_actions=valid_actions,
        metadata={
            "task_objective": task_objective,
            "difficulty": difficulty,
            "step": step_index,
            "deterministic_task": True,
        },
    )


def build_agent_tasks() -> list[AgentTask]:
    easy_objective = "Enter a long position when momentum is clearly bullish and BUY is valid."
    medium_objective = "Exit an overheated long position to lock gains when SELL is valid."
    hard_objective = "Trade a full reversal sequence: buy the breakout, hold the trend, then sell the reversal."
    return [
        AgentTask(
            name="bullish_entry",
            difficulty="easy",
            objective=easy_objective,
            steps=(
                AgentTaskStep(
                    observation=make_observation(
                        current_price=103.2,
                        price_window=[99.8, 100.4, 100.9, 101.7, 102.1, 102.8, 103.2],
                        balance=10_000.0,
                        shares_held=0,
                        portfolio_value=10_000.0,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY],
                        sma_short=102.5,
                        sma_long=101.4,
                        rsi=58.0,
                        task_objective=easy_objective,
                        step_index=1,
                        difficulty="easy",
                    ),
                    expected_action=TradingActionType.BUY,
                ),
            ),
        ),
        AgentTask(
            name="overbought_exit",
            difficulty="medium",
            objective=medium_objective,
            steps=(
                AgentTaskStep(
                    observation=make_observation(
                        current_price=108.6,
                        price_window=[103.1, 104.2, 105.5, 106.7, 107.9, 108.2, 108.6],
                        balance=9_891.2,
                        shares_held=1,
                        portfolio_value=9_999.8,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY, TradingActionType.SELL],
                        sma_short=107.9,
                        sma_long=105.8,
                        rsi=83.0,
                        task_objective=medium_objective,
                        step_index=1,
                        difficulty="medium",
                    ),
                    expected_action=TradingActionType.SELL,
                ),
                AgentTaskStep(
                    observation=make_observation(
                        current_price=107.8,
                        price_window=[104.2, 105.5, 106.7, 107.9, 108.2, 108.6, 107.8],
                        balance=9_999.2,
                        shares_held=0,
                        portfolio_value=9_999.2,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY],
                        sma_short=108.0,
                        sma_long=106.4,
                        rsi=71.0,
                        task_objective=medium_objective,
                        step_index=2,
                        difficulty="medium",
                    ),
                    expected_action=TradingActionType.HOLD,
                ),
            ),
        ),
        AgentTask(
            name="trend_reversal_sequence",
            difficulty="hard",
            objective=hard_objective,
            steps=(
                AgentTaskStep(
                    observation=make_observation(
                        current_price=100.9,
                        price_window=[98.7, 99.1, 99.6, 100.1, 100.4, 100.7, 100.9],
                        balance=10_000.0,
                        shares_held=0,
                        portfolio_value=10_000.0,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY],
                        sma_short=100.4,
                        sma_long=99.7,
                        rsi=56.0,
                        task_objective=hard_objective,
                        step_index=1,
                        difficulty="hard",
                    ),
                    expected_action=TradingActionType.BUY,
                ),
                AgentTaskStep(
                    observation=make_observation(
                        current_price=102.4,
                        price_window=[99.6, 100.1, 100.4, 100.7, 100.9, 101.8, 102.4],
                        balance=9_899.0,
                        shares_held=1,
                        portfolio_value=10_001.4,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY, TradingActionType.SELL],
                        sma_short=101.2,
                        sma_long=100.3,
                        rsi=63.0,
                        task_objective=hard_objective,
                        step_index=2,
                        difficulty="hard",
                    ),
                    expected_action=TradingActionType.HOLD,
                ),
                AgentTaskStep(
                    observation=make_observation(
                        current_price=103.1,
                        price_window=[100.4, 100.7, 100.9, 101.8, 102.4, 102.8, 103.1],
                        balance=9_899.0,
                        shares_held=1,
                        portfolio_value=10_002.1,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY, TradingActionType.SELL],
                        sma_short=102.2,
                        sma_long=101.0,
                        rsi=67.0,
                        task_objective=hard_objective,
                        step_index=3,
                        difficulty="hard",
                    ),
                    expected_action=TradingActionType.HOLD,
                ),
                AgentTaskStep(
                    observation=make_observation(
                        current_price=101.6,
                        price_window=[100.9, 101.8, 102.4, 102.8, 103.1, 102.2, 101.6],
                        balance=9_899.0,
                        shares_held=1,
                        portfolio_value=10_000.6,
                        valid_actions=[TradingActionType.HOLD, TradingActionType.BUY, TradingActionType.SELL],
                        sma_short=102.4,
                        sma_long=102.5,
                        rsi=76.0,
                        task_objective=hard_objective,
                        step_index=4,
                        difficulty="hard",
                    ),
                    expected_action=TradingActionType.SELL,
                ),
            ),
        ),
    ]


def grade_task(expected_actions: list[TradingActionType], chosen_actions: list[TradingActionType]) -> float:
    if not expected_actions or len(expected_actions) != len(chosen_actions):
        return 0.0
    correct = sum(1 for expected, chosen in zip(expected_actions, chosen_actions) if expected == chosen)
    return correct / len(expected_actions)


def run_agent_tasks() -> list[tuple[AgentTask, float, list[TradingActionType]]]:
    client = (
        OpenAI(base_url=inference.API_BASE_URL, api_key=inference.API_KEY)
        if inference.API_BASE_URL and inference.API_KEY and inference.MODEL_NAME
        else None
    )
    results: list[tuple[AgentTask, float, list[TradingActionType]]] = []
    for task in build_agent_tasks():
        chosen_actions: list[TradingActionType] = []
        observed_prices: list[float] = []
        for step_index, task_step in enumerate(task.steps, start=1):
            chosen_action = inference.choose_action(client, step_index, task_step.observation, observed_prices)
            chosen_actions.append(chosen_action)
            observed_prices.append(float(task_step.observation.current_price))
        score = grade_task(
            expected_actions=[step.expected_action for step in task.steps],
            chosen_actions=chosen_actions,
        )
        results.append((task, score, chosen_actions))
    return results


def main() -> None:
    results = run_agent_tasks()
    for task, score, chosen_actions in results:
        actions = ",".join(action.value for action in chosen_actions)
        print(
            f"[TASK] name={task.name} difficulty={task.difficulty} score={score:.2f} actions={actions}",
            flush=True,
        )
    overall = sum(score for _, score, _ in results) / max(len(results), 1)
    print(f"[TASK_SUMMARY] count={len(results)} overall_score={overall:.2f}", flush=True)


if __name__ == "__main__":
    main()
