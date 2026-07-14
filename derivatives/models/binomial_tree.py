"""二叉树期权定价。"""

from __future__ import annotations

import math

import numpy as np


def european_call_tree(S: float, K: float, r: float, sigma: float, T: float, steps: int) -> float:
    """欧式看涨期权 CRR 二叉树定价。"""
    u = np.exp(sigma * np.sqrt(T / steps))
    d = 1.0 / u
    p = (np.exp(r * T / steps) - d) / (u - d)

    prices = np.zeros(steps + 1)
    values = np.zeros(steps + 1)
    prices[0] = S * d**steps
    values[0] = max(prices[0] - K, 0.0)

    for i in range(1, steps + 1):
        prices[i] = prices[i - 1] * (u**2)
        values[i] = max(prices[i] - K, 0.0)

    for j in range(steps, 0, -1):
        for i in range(j):
            values[i] = (p * values[i + 1] + (1 - p) * values[i]) / np.exp(r * T / steps)

    return float(values[0])


def european_put_tree(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
) -> float:
    """欧式看跌期权二叉树定价（含连续股息率）。"""
    u = np.exp(sigma * np.sqrt(T / steps))
    d = 1.0 / u
    p = (np.exp((r - q) * T / steps) - d) / (u - d)
    option_values = [max(K - S * u ** (steps - 2 * i), 0.0) for i in range(steps + 1)]

    for j in range(steps, 0, -1):
        for i in range(j):
            option_values[i] = (p * option_values[i] + (1 - p) * option_values[i + 1]) * np.exp(
                -r * T / steps
            )

    return float(option_values[0])


def american_option_tree(
    option_type: str,
    steps: int,
    S0: float,
    T: float,
    sigma: float,
    K: float,
    r: float,
    b: float,
) -> float:
    """美式期权二叉树定价。

    Parameters
    ----------
    option_type : str
        ``'C'`` 或 ``'P'``。
    b : float
        持有成本；``b=r`` 为无股利，``b=r-q`` 为支付股利，``b=0`` 为期货期权。
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(b * dt) - d) / (u - d)

    prices = np.zeros((steps + 1, steps + 1))
    prices[0, 0] = S0
    for i in range(1, steps + 1):
        for j in range(i):
            prices[j, i] = prices[j, i - 1] * u
            prices[j + 1, i] = prices[j, i - 1] * d

    if option_type.upper() == "C":
        intrinsic = np.maximum(prices - K, 0.0)
    else:
        intrinsic = np.maximum(K - prices, 0.0)

    values = np.zeros_like(prices)
    values[:, -1] = intrinsic[:, -1]

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = (values[j, i + 1] * p + values[j + 1, i + 1] * (1 - p)) / math.exp(r * dt)
            values[j, i] = max(continuation, intrinsic[j, i])

    return float(values[0, 0])
