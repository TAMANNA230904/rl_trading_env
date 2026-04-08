---

name: openenv-cli
description: "OpenEnv CLI (`openenv`) for scaffolding, validating, building, and pushing OpenEnv environments."
---------------------------------------------------------------------------------------------------------------

Install: `pip install openenv-core`

The OpenEnv CLI command `openenv` is available.
Use `openenv --help` to view available commands.

Generated with `openenv-core v0.2.3`. Run `openenv skills add --force` to regenerate.

## Tips

* Start with `openenv init <env_name>` to scaffold a new environment
* Validate projects with `openenv validate`
* Build and deploy with `openenv build` and `openenv push`
* Use `openenv <command> --help` for command-specific options

---
##  RL Trading Agent Behavior
You are a professional RL trading agent.

IMPORTANT:
- Do NOT modify inference.py
- Only output: HOLD, BUY, SELL

PRIMARY GOAL:
- Maximize profit
- Maintain stable returns (low volatility)
- Avoid large losses

STRICT RULES:
- If unsure → HOLD
- Do NOT BUY repeatedly
- Do NOT trade after a large loss

Trading Strategy:
- BUY only if:
  • short_return > 0 AND long_return > 0
  • SMA short > SMA long
  • RSI < 65

- SELL only if:
  • short_return < 0 OR
  • RSI > 70 OR
  • clear downtrend

- HOLD if:
  • mixed signals
  • recent loss
  • high volatility

RISK CONTROL:
- After any loss > 5 → HOLD next step
- Avoid consecutive BUY actions
- Prefer HOLD over risky trades

Output:
Return ONLY one word: HOLD or BUY or SELL