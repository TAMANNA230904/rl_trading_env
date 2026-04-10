---
title: Meta AI Trading Env
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# RL Trading Environment

A single-asset reinforcement learning trading environment for OpenEnv with three benchmark task categories: easy, medium, and hard.

## Environment

The environment exposes a discrete action space with:
- `HOLD`
- `BUY`
- `SELL`

Each episode generates a synthetic price series, maintains account state, enforces trading constraints, and returns observations suitable for policy learning.

## Benchmark Tasks

The benchmark uses predefined task scenarios stored in [`tasks/`](C:\Users\Lenovo\Desktop\meta_ai\tasks):
- `task_easy`: upward-trending market with mild pullbacks
- `task_medium`: breakout-prone market with sharper reversals
- `task_hard`: noisy whipsaw market with regime flips

`inference.py` runs all three tasks autonomously and reports per-task totals plus an overall summary.

## Local Run

Build the image:

```bash
docker build -t openenv-rl_trading .
```

Run inference:

```bash
python -u inference.py
```

Validate the project:

```bash
openenv validate
```

## Runtime

The Docker container serves the OpenEnv app on port `8000`.

For hackathon submissions, the inference runner must use the injected LiteLLM proxy variables:

```python
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)
```

Do not rely on `HF_TOKEN`, hardcoded credentials, or another provider for submitted builds.
