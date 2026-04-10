"""Easy benchmark scenarios for the trading environment."""

TASK = {
    "id": "task_easy",
    "difficulty": "easy",
    "name": "Trend Follow Basic",
    "description": "A mostly upward-trending market with mild pullbacks. Follow obvious direction and avoid unnecessary churn.",
    "market_mode": "trend_up",
    "drift": 0.0014,
    "volatility": 0.0060,
    "seasonal_amplitude": 0.0015,
    "shock_scale": 0.0020,
    "reward_scale": 1.1,
    "target_steps": 16,
    "seed": 101,
}
