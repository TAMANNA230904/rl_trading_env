# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the RL trading environment."""

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TradingActionType(str, Enum):
    """Discrete trading actions supported by the environment."""

    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"


class RewardMode(str, Enum):
    """Reward shaping modes available in the environment metadata."""

    PORTFOLIO_DELTA = "portfolio_delta"
    SHARPE_LIKE = "sharpe_like"
    SORTINO_LIKE = "sortino_like"


class RlTradingAction(Action):
    """Discrete trading action."""

    action: TradingActionType = Field(
        default=TradingActionType.HOLD,
        description="Discrete action: HOLD, BUY, or SELL.",
    )


class RlTradingObservation(Observation):
    """Observation emitted by the trading environment."""

    current_price: float = Field(..., description="Current asset price.")
    price_window: list[float] = Field(
        default_factory=list,
        description="Rolling normalized or raw price window ending at current_price.",
    )
    balance: float = Field(..., description="Available cash balance.")
    shares_held: int = Field(..., description="Current inventory.")
    portfolio_value: float = Field(..., description="Cash plus mark-to-market inventory.")
    sma_short: float | None = Field(
        default=None,
        description="Optional short moving average of recent prices.",
    )
    sma_long: float | None = Field(
        default=None,
        description="Optional long moving average of recent prices.",
    )
    rsi: float | None = Field(
        default=None,
        description="Optional Relative Strength Index indicator.",
    )
    valid_actions: list[TradingActionType] = Field(
        default_factory=list,
        description="Actions that are executable given current balance and holdings.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic information including fills, costs, and reward components.",
    )
