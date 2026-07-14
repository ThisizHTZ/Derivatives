"""障碍期权定价。"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def double_barrier_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    upper: float,
    lower: float,
    max_iterations: int = 100,
) -> float:
    """双障碍看涨期权（级数解法）。"""
    dt = sigma * np.sqrt(T)
    price = 0.0

    for n in range(-max_iterations, max_iterations + 1):
        factor_upper = (upper / S) ** (2 * n)
        factor_lower = (lower / S) ** (2 * n)
        d1 = (np.log(S * factor_upper / K) + (r + 0.5 * sigma**2) * T) / dt
        d2 = d1 - dt
        d3 = (np.log(S * factor_lower / K) + (r + 0.5 * sigma**2) * T) / dt
        d4 = d3 - dt
        price += factor_upper * (norm.cdf(d1) - norm.cdf(d2))
        price -= factor_lower * (norm.cdf(-d3) - norm.cdf(-d4))

    return np.exp(-r * T) * price


def double_barrier_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    upper: float,
    lower: float,
    max_iterations: int = 100,
) -> float:
    """双障碍看跌期权（级数解法）。"""
    dt = sigma * np.sqrt(T)
    price = 0.0

    for n in range(-max_iterations, max_iterations + 1):
        factor_upper = (upper / S) ** (2 * n)
        factor_lower = (lower / S) ** (2 * n)
        d1 = (np.log(S * factor_upper / K) + (r + 0.5 * sigma**2) * T) / dt
        d2 = d1 - dt
        d3 = (np.log(S * factor_lower / K) + (r + 0.5 * sigma**2) * T) / dt
        d4 = d3 - dt
        price += factor_upper * (norm.cdf(-d1) - norm.cdf(-d2))
        price -= factor_lower * (norm.cdf(d3) - norm.cdf(d4))

    return max(np.exp(-r * T) * price, 0.0)


def barrier_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    upper: float,
    lower: float,
    dividend: float,
    max_iterations: int = 50,
) -> float:
    """含股息率的双向障碍看涨期权。"""
    b = r - dividend
    e_level = lower * np.exp(dividend * T)
    sum1 = 0.0
    sum2 = 0.0

    for n in range(-max_iterations, max_iterations):
        upper_n = upper * (lower / upper) ** (2 * n)
        lower_n = lower * (upper / lower) ** (2 * n)
        d1 = (np.log(S * upper_n / (e_level * lower_n)) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        d3 = (np.log(lower_n**2 / (S * upper_n * K)) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d4 = d3 - sigma * np.sqrt(T)
        sum1 += (upper_n / lower_n) ** n * (lower / S) ** (2 * n) * (norm.cdf(d1) - norm.cdf(d2))
        sum2 += (lower_n / upper_n) ** (n + 1) * (upper_n / S) ** (2 * n) * (norm.cdf(-d3) - norm.cdf(-d4))

    return S * np.exp((b - r) * T) * sum1 - K * np.exp(-r * T) * sum2


def binary_barrier_option(
    S: float,
    X: float,
    H: float,
    K: float,
    r: float,
    b: float,
    T: float,
    sigma: float,
    phi: int,
    eta: int,
) -> float:
    """二元障碍期权（向下敲出现金或无看跌示例分支）。"""
    mu = (b - sigma**2 / 2) / sigma**2
    lambda_ = np.sqrt(mu**2 + (2 * r) / sigma**2)

    x1 = np.log(S / X) / (sigma * np.sqrt(T)) + (mu + 1) * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (mu + 1) * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (S * X)) / (sigma * np.sqrt(T)) + (mu + 1) * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (mu + 1) * sigma * np.sqrt(T)

    b1 = K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
    b2 = K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
    b3 = K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
    b4 = K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))

    return b1 - b2 + b3 - b4
