"""OpenEnv trading simulator for reinforcement learning and policy inference."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from tasks import TASK_REGISTRY

from rl_trading_env.models import (
    RewardMode,
    RlTradingAction,
    RlTradingObservation,
    TradingActionType,
)


@dataclass(slots=True)
class TradeExecution:
    """Execution details for a single action."""

    action: TradingActionType
    executed: bool
    quantity: int
    transaction_cost: float
    invalid_reason: str | None = None


class RlTradingEnvironment(Environment):
    """
    Single-asset trading simulator with constrained discrete market actions.

    Each episode produces a synthetic price trajectory and exposes an RL-ready
    observation consisting of the current price, a rolling history window,
    available cash, current holdings, total portfolio value, and optional
    technical indicators such as SMA and RSI. Agents interact through
    discrete `HOLD`, `BUY`, and `SELL` actions, with invalid trades blocked
    when balance or inventory constraints are not satisfied.

    Rewards are derived from portfolio value changes and therefore reflect
    market movement, position exposure, and transaction costs. Optional
    Sharpe-like and Sortino-like shaping can be enabled for risk-adjusted
    training or evaluation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        episode_length: int = 256,
        window_size: int = 20,
        initial_balance: float = 10_000.0,
        trade_size: int = 1,
        transaction_cost_pct: float = 0.001,
        include_sma: bool = True,
        include_rsi: bool = True,
        sma_short_window: int = 5,
        sma_long_window: int = 14,
        rsi_window: int = 14,
        reward_mode: RewardMode = RewardMode.PORTFOLIO_DELTA,
        risk_penalty_weight: float = 0.02,
        invalid_action_penalty: float = 0.001,
        reward_scale: float = 1.00,
        seed: int = 7,
    ):
        self.episode_length = max(episode_length, window_size + 2)
        self.window_size = max(2, window_size)
        self.initial_balance = float(initial_balance)
        self.trade_size = max(1, trade_size)
        self.transaction_cost_pct = max(0.0, transaction_cost_pct)
        self.include_sma = include_sma
        self.include_rsi = include_rsi
        self.sma_short_window = max(2, sma_short_window)
        self.sma_long_window = max(self.sma_short_window + 1, sma_long_window)
        self.rsi_window = max(2, rsi_window)
        self.reward_mode = reward_mode
        self.risk_penalty_weight = max(0.0, risk_penalty_weight)
        self.invalid_action_penalty = max(0.0, invalid_action_penalty)
        self.reward_scale = max(1.0, reward_scale)
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._reset_count = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prices = np.array([], dtype=np.float64)
        self._t = 0
        self._balance = self.initial_balance
        self._shares_held = 0
        self._portfolio_history: list[float] = []
        self._returns_history: list[float] = []
        self._task_id = "task_easy"
        self._task = TASK_REGISTRY[self._task_id]

    def reset(self, task_id: str | None = None, seed: int | None = None) -> RlTradingObservation:
        """Reset the episode and return the initial observation."""
        self._reset_count += 1
        self._task_id, self._task = self._resolve_task(task_id)
        active_seed = self._task["seed"] if seed is None else seed
        self._rng = np.random.default_rng(active_seed)
        self._prices = self._generate_price_series()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._t = self.window_size - 1
        self._balance = self.initial_balance
        self._shares_held = 0
        portfolio_value = self._portfolio_value(self._current_price())
        self._portfolio_history = [portfolio_value]
        self._returns_history = []
        return self._build_observation(
            reward=0.0,
            done=False,
            trade=TradeExecution(
                action=TradingActionType.HOLD,
                executed=True,
                quantity=0,
                transaction_cost=0.0,
            ),
        )

    def step(self, action: RlTradingAction) -> RlTradingObservation:  # type: ignore[override]
        """Execute a single trading step."""
        current_price = self._current_price()
        previous_value = self._portfolio_value(current_price)
        trade = self._execute_trade(action.action, current_price)

        self._state.step_count += 1
        self._t = min(self._t + 1, len(self._prices) - 1)

        next_price = self._current_price()
        current_value = self._portfolio_value(next_price)
        portfolio_delta = current_value - previous_value
        step_return = portfolio_delta / max(previous_value, 1e-8)
        self._portfolio_history.append(current_value)
        self._returns_history.append(step_return)

        reward = self._compute_reward(
            portfolio_delta=portfolio_delta,
            step_return=step_return,
            trade=trade,
        )
        done = self._t >= len(self._prices) - 1
        return self._build_observation(reward=reward, done=done, trade=trade)

    @property
    def state(self) -> State:
        """Return the shared OpenEnv state."""
        return self._state

    def _generate_price_series(self) -> np.ndarray:
        """Create a task-shaped price path for one episode."""
        total_steps = self.episode_length + self.window_size
        drift = float(self._task["drift"])
        volatility = float(self._task["volatility"])
        seasonal_amplitude = float(self._task["seasonal_amplitude"])
        shock_scale = float(self._task["shock_scale"])
        market_mode = str(self._task["market_mode"])
        shocks = self._rng.normal(loc=drift, scale=volatility, size=total_steps)
        prices = np.empty(total_steps, dtype=np.float64)
        prices[0] = 100.0
        for idx in range(1, total_steps):
            seasonal = seasonal_amplitude * np.sin(idx / 14.0)
            regime = self._regime_adjustment(idx, market_mode)
            impulse = self._shock_adjustment(idx, market_mode, shock_scale)
            prices[idx] = max(1.0, prices[idx - 1] * (1.0 + shocks[idx] + seasonal + regime + impulse))
        return prices

    def _execute_trade(self, action: TradingActionType, price: float) -> TradeExecution:
        """Apply a discrete trade subject to cash and inventory constraints."""
        if action == TradingActionType.HOLD:
            return TradeExecution(action=action, executed=True, quantity=0, transaction_cost=0.0)

        quantity = self.trade_size
        gross_value = price * quantity
        transaction_cost = gross_value * self.transaction_cost_pct

        if action == TradingActionType.BUY:
            total_cost = gross_value + transaction_cost
            if self._balance + 1e-8 < total_cost:
                return TradeExecution(
                    action=action,
                    executed=False,
                    quantity=0,
                    transaction_cost=0.0,
                    invalid_reason="Insufficient balance for BUY action.",
                )
            self._balance -= total_cost
            self._shares_held += quantity
            return TradeExecution(
                action=action,
                executed=True,
                quantity=quantity,
                transaction_cost=transaction_cost,
            )

        if self._shares_held < quantity:
            return TradeExecution(
                action=action,
                executed=False,
                quantity=0,
                transaction_cost=0.0,
                invalid_reason="Insufficient holdings for SELL action.",
            )

        net_proceeds = gross_value - transaction_cost
        self._shares_held -= quantity
        self._balance += net_proceeds
        return TradeExecution(
            action=action,
            executed=True,
            quantity=quantity,
            transaction_cost=transaction_cost,
        )

    def _compute_reward(
        self,
        portfolio_delta: float,
        step_return: float,
        trade: TradeExecution,
    ) -> float:
        """Reward based on portfolio change with optional risk shaping."""
        normalized_delta = portfolio_delta / max(self.initial_balance, 1e-8)
        reward = normalized_delta

        if self.reward_mode == RewardMode.SHARPE_LIKE:
            reward = self._risk_adjusted_reward(step_return, downside_only=False)
        elif self.reward_mode == RewardMode.SORTINO_LIKE:
            reward = self._risk_adjusted_reward(step_return, downside_only=True)

        if trade.action == TradingActionType.HOLD:
            reward -= 0.00005

        if not trade.executed and trade.action != TradingActionType.HOLD:
            reward -= self.invalid_action_penalty

        scaled_reward = 0.5 + (reward * self.reward_scale * 100.0)
        bounded_reward = min(max(scaled_reward, 0.0), 0.999)
        return float(bounded_reward)

    def _resolve_task(self, task_id: str | None) -> tuple[str, dict]:
        resolved_task_id = task_id or self._task_id
        if resolved_task_id not in TASK_REGISTRY:
            resolved_task_id = "task_easy"
        task = TASK_REGISTRY[resolved_task_id]
        self.reward_scale = max(1.0, float(task["reward_scale"]))
        return resolved_task_id, task

    def _regime_adjustment(self, idx: int, market_mode: str) -> float:
        if market_mode == "trend_up":
            return 0.0006 if idx > self.window_size else 0.0
        if market_mode == "momentum_breakout":
            cycle = idx % 24
            if 8 <= cycle <= 12:
                return 0.0020
            if cycle >= 18:
                return -0.0009
            return 0.0
        if market_mode == "whipsaw_reversal":
            cycle = idx % 20
            if cycle < 4:
                return 0.0022
            if 4 <= cycle < 8:
                return -0.0026
        return 0.0

    def _shock_adjustment(self, idx: int, market_mode: str, shock_scale: float) -> float:
        if market_mode == "trend_up":
            return 0.0
        if market_mode == "momentum_breakout" and idx % 17 == 0:
            return shock_scale
        if market_mode == "whipsaw_reversal":
            if idx % 9 == 0:
                return shock_scale
            if idx % 11 == 0:
                return -shock_scale
        return 0.0

    def _risk_adjusted_reward(self, step_return: float, downside_only: bool) -> float:
        """Approximate Sharpe/Sortino reward shaping from realized returns."""
        returns = np.asarray(self._returns_history, dtype=np.float64)
        if returns.size == 0:
            return float(step_return)

        excess_mean = float(np.mean(returns))
        if downside_only:
            risk_sample = returns[returns < 0.0]
            dispersion = float(np.std(risk_sample)) if risk_sample.size else 0.0
        else:
            dispersion = float(np.std(returns))

        risk_term = dispersion if dispersion > 1e-8 else 1e-8
        shaped = excess_mean / risk_term
        return float(step_return + self.risk_penalty_weight * shaped)

    def _build_observation(
        self,
        reward: float,
        done: bool,
        trade: TradeExecution,
    ) -> RlTradingObservation:
        """Assemble the latest observation and diagnostics."""
        current_price = self._current_price()
        price_window = self._window_prices()
        valid_actions = self._valid_actions(current_price)
        metadata = {
            "task_id": self._task_id,
            "difficulty": self._task["difficulty"],
            "scenario_name": self._task["name"],
            "scenario_description": self._task["description"],
            "market_mode": self._task["market_mode"],
            "executed_action": trade.action.value,
            "trade_executed": trade.executed,
            "filled_quantity": trade.quantity,
            "transaction_cost": trade.transaction_cost,
            "reward_mode": self.reward_mode.value,
            "reward_scale": self.reward_scale,
            "step": self._state.step_count,
        }
        if trade.invalid_reason:
            metadata["invalid_reason"] = trade.invalid_reason

        return RlTradingObservation(
            current_price=float(current_price),
            price_window=[float(price) for price in price_window],
            balance=float(self._balance),
            shares_held=int(self._shares_held),
            portfolio_value=float(self._portfolio_value(current_price)),
            sma_short=self._sma(self.sma_short_window) if self.include_sma else None,
            sma_long=self._sma(self.sma_long_window) if self.include_sma else None,
            rsi=self._rsi(self.rsi_window) if self.include_rsi else None,
            valid_actions=valid_actions,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _valid_actions(self, price: float) -> list[TradingActionType]:
        """Return the currently executable actions."""
        valid = [TradingActionType.HOLD]
        total_buy_cost = (price * self.trade_size) * (1.0 + self.transaction_cost_pct)
        if self._balance + 1e-8 >= total_buy_cost:
            valid.append(TradingActionType.BUY)
        if self._shares_held >= self.trade_size:
            valid.append(TradingActionType.SELL)
        return valid

    def _window_prices(self) -> np.ndarray:
        """Return the rolling lookback window ending at the current index."""
        start = max(0, self._t - self.window_size + 1)
        window = self._prices[start : self._t + 1]
        if len(window) < self.window_size:
            pad = np.full(self.window_size - len(window), window[0], dtype=np.float64)
            window = np.concatenate([pad, window])
        return window

    def _sma(self, length: int) -> float:
        """Simple moving average over the latest prices."""
        window = self._window_prices()
        use = window[-min(length, len(window)) :]
        return float(np.mean(use))

    def _rsi(self, length: int) -> float:
        """Relative Strength Index over recent prices."""
        window = self._window_prices()
        deltas = np.diff(window[-min(length + 1, len(window)) :])
        if deltas.size == 0:
            return 50.0
        gains = np.clip(deltas, a_min=0.0, a_max=None)
        losses = -np.clip(deltas, a_min=None, a_max=0.0)
        avg_gain = float(np.mean(gains)) if gains.size else 0.0
        avg_loss = float(np.mean(losses)) if losses.size else 0.0
        if avg_loss <= 1e-8:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _portfolio_value(self, price: float) -> float:
        """Mark-to-market portfolio value."""
        return self._balance + (self._shares_held * price)

    def _current_price(self) -> float:
        """Current price at the active timestep."""
        return float(self._prices[self._t])


if __name__ == "__main__":
    env = RlTradingEnvironment()
    obs = env.reset(task_id="task_easy")
    print("Reset observation:", obs.model_dump())
    for action in [
        TradingActionType.BUY,
        TradingActionType.HOLD,
        TradingActionType.SELL,
    ]:
        step_obs = env.step(RlTradingAction(action=action))
        print(f"{action.value} -> reward={step_obs.reward:.6f}, value={step_obs.portfolio_value:.2f}")
