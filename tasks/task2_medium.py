"""Medium benchmark scenarios for the trading environment."""

TASK = {
    "id": "task_medium",
    "difficulty": "medium",
    "name": "Momentum Confirmation",
    "description": "A breakout-prone market with sharper reversals. Use momentum confirmation from indicators before entering or exiting.",
    "market_mode": "momentum_breakout",
    "drift": 0.0009,
    "volatility": 0.0100,
    "seasonal_amplitude": 0.0024,
    "shock_scale": 0.0045,
    "reward_scale": 1.0,
    "target_steps": 32,
    "seed": 202,
}
