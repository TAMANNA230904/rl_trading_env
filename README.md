---
title: Meta AI Trading Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# RL Trading Environment

A complete single-asset reinforcement learning trading environment for OpenEnv.

The environment exposes a discrete action space with:
- `HOLD`
- `BUY`
- `SELL`

Each episode generates a synthetic price series, maintains account state, enforces
trading constraints, and returns observations suitable for policy learning.

## Observation Space

Each observation includes:
- `current_price`
- `price_window`: rolling window of recent prices
- `balance`
- `shares_held`
- `portfolio_value`
- `valid_actions`
- Optional `sma_short`, `sma_long`, and `rsi`

The observation metadata also reports trade execution diagnostics such as
filled quantity, transaction costs, reward mode, and invalid-action reasons.

## Action Space

Actions are represented by `RlTradingAction(action=...)` where `action` is one of:
- `TradingActionType.HOLD`
- `TradingActionType.BUY`
- `TradingActionType.SELL`

Constraint handling:
- `BUY` is rejected when balance cannot cover the asset plus transaction cost
- `SELL` is rejected when holdings are insufficient
- Invalid actions remain observable via `valid_actions` and incur a configurable penalty

## Reward Function

The default reward is normalized portfolio value change:

```text
reward = (portfolio_value_t - portfolio_value_t-1) / initial_balance
```

This naturally includes:
- Realized and unrealized PnL
- Transaction costs
- Mark-to-market inventory changes

Optional reward shaping modes:
- `portfolio_delta`
- `sharpe_like`
- `sortino_like`

The risk-adjusted modes use realized returns history to shape reward with a
Sharpe-like or downside-risk-aware term.

## Environment Parameters

`RlTradingEnvironment` supports the following main configuration knobs:
- `episode_length`
- `window_size`
- `initial_balance`
- `trade_size`
- `transaction_cost_pct`
- `include_sma`
- `include_rsi`
- `sma_short_window`
- `sma_long_window`
- `rsi_window`
- `reward_mode`
- `risk_penalty_weight`
- `invalid_action_penalty`
- `seed`

## Quick Start

```python
from rl_trading_env import RlTradingAction, RlTradingEnv, TradingActionType

with RlTradingEnv(base_url="http://localhost:8000") as env:
    reset_result = env.reset()
    print(reset_result.observation.portfolio_value)

    step_result = env.step(RlTradingAction(action=TradingActionType.BUY))
    print(step_result.observation.current_price)
    print(step_result.observation.valid_actions)
    print(step_result.reward)
```

## Local Development

Run the core environment directly:

```bash
python server/rl_trading_env_environment.py
```

Run the server:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
openenv validate
```

Build the Docker image:

```bash
openenv build
```
