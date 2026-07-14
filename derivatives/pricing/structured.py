"""结构化衍生品定价。"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from derivatives.pricing.bsm import black_scholes_merton


def shark_fin_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    upper_barrier: float,
    steps: int,
    simulations: int,
    option_type: str = "call",
    seed: int | None = None,
) -> float:
    """单向鲨鱼鳍期权蒙特卡洛定价。"""
    rng = np.random.default_rng(seed)
    dt = T / steps
    payoffs = np.zeros(simulations)

    for i in range(simulations):
        shocks = rng.normal(size=steps)
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
        prices = S * np.exp(np.cumsum(log_returns))
        if np.max(prices) >= upper_barrier:
            if option_type == "call":
                payoffs[i] = max(prices[-1] - K, 0.0)
            else:
                payoffs[i] = max(K - prices[-1], 0.0)

    return float(np.exp(-r * T) * np.mean(payoffs))


def up_and_out_call(
    S: float,
    X: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    H: float,
    K: float,
    participate_rate: float,
    floor_return: float,
) -> float:
    """向上敲出看涨鲨鱼鳍结构。"""
    b = r - q
    mu = (b - 0.5 * sigma**2) / sigma**2
    lamb = math.sqrt(mu**2 + (2 * r / (sigma**2)))

    x1 = math.log(S / X) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    x2 = math.log(S / H) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y1 = math.log(H**2 / (S * X)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y2 = math.log(H / S) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    z = math.log(H / S) / (sigma * math.sqrt(T)) + lamb * sigma * math.sqrt(T)

    a = S * math.exp((b - r) * T) * norm.cdf(x1) - X * math.exp(-r * T) * norm.cdf(x1 - sigma * math.sqrt(T))
    b_term = S * math.exp((b - r) * T) * norm.cdf(x2) - X * math.exp(-r * T) * norm.cdf(x2 - sigma * math.sqrt(T))
    c = S * math.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y1) - X * math.exp(-r * T) * (
        H / S
    ) ** (2 * mu) * norm.cdf(-y1 + sigma * math.sqrt(T))
    d = S * math.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y2) - X * math.exp(-r * T) * (
        H / S
    ) ** (2 * mu) * norm.cdf(-y2 + sigma * math.sqrt(T))
    f = K * (
        (H / S) ** (mu + lamb) * norm.cdf(-z)
        + (H / S) ** (mu - lamb) * norm.cdf(-z + 2 * lamb * sigma * math.sqrt(T))
    )

    price = max(a - b_term + c - d + f, floor_return) * participate_rate
    return price


def _down_and_in_put(S: float, K: float, H: float, T: float, r: float, sigma: float) -> float:
    lambda_ = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(H / S) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)

    return (
        K * np.exp(-r * T) * norm.cdf(y)
        - S * norm.cdf(y - sigma * np.sqrt(T))
        - K * np.exp(-r * T) * (H / S) ** (2 * lambda_) * norm.cdf(y1)
        + S * (H / S) ** (2 * lambda_) * norm.cdf(y1 - sigma * np.sqrt(T))
    )


def airbag_option_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    participate_rate: float,
) -> float:
    """安全气囊结构：香草看涨 + 向下敲入看跌，按参与率缩放。"""
    vanilla_call = black_scholes_merton(S, K, T, r, sigma, option_type="call")
    barrier_put = _down_and_in_put(S, K, H, T, r, sigma)
    return (vanilla_call + barrier_put) * participate_rate


def range_accrual_price(
    S0: float,
    T: float,
    sigma: float,
    r: float,
    lower_bound: float,
    upper_bound: float,
    trading_days: int = 240,
    paths: int = 1000,
    seed: int | None = None,
) -> float:
    """Range Accrual 期权蒙特卡洛定价。"""
    rng = np.random.default_rng(seed)
    dt = T / trading_days
    payoffs = []

    for _ in range(paths):
        shocks = rng.standard_normal(trading_days)
        brownian = np.cumsum(shocks) * np.sqrt(dt)
        t_grid = np.linspace(0.0, T, trading_days)
        log_price = (r - 0.5 * sigma**2) * t_grid + sigma * brownian
        prices = S0 * np.exp(log_price)
        in_range_days = np.sum((prices > lower_bound) & (prices < upper_bound))
        payoffs.append(in_range_days)

    return float(np.exp(-r * T) * np.mean(payoffs))
