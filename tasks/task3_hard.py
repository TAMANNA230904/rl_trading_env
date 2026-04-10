"""Hard benchmark scenarios for the trading environment."""

TASK = {
    "id": "task_hard",
    "difficulty": "hard",
    "name": "Whipsaw Survival",
    "description": "A choppy market with regime flips and noisy reversals. Preserve capital and avoid overtrading under uncertainty.",
    "market_mode": "whipsaw_reversal",
    "drift": 0.0002,
    "volatility": 0.0140,
    "seasonal_amplitude": 0.0032,
    "shock_scale": 0.0075,
    "reward_scale": 0.95,
    "target_steps": 48,
    "seed": 303,
}
