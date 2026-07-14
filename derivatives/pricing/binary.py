"""二元期权与缺口类期权定价。"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, T: float, b: float, sigma: float) -> float:
    return (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def binary_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """现金结算型欧式二元期权（到期 ITM 支付 1 单位现金）。"""
    d = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return np.exp(-r * T) * norm.cdf(d)
    if option_type == "put":
        return np.exp(-r * T) * norm.cdf(-d)
    raise ValueError("option_type must be 'call' or 'put'")


def cash_or_nothing_call(
    S: float,
    K: float,
    cash: float,
    T: float,
    r: float,
    b: float,
    sigma: float,
) -> float:
    """现金或无看涨期权。"""
    d = (np.log(S / K) + (b - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return cash * np.exp(-r * T) * norm.cdf(d)


def cash_or_nothing_put(
    S: float,
    K: float,
    cash: float,
    T: float,
    r: float,
    b: float,
    sigma: float,
) -> float:
    """现金或无看跌期权。"""
    d = (np.log(S / K) + (b - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return cash * np.exp(-r * T) * norm.cdf(-d)


def asset_or_nothing_call(
    S: float,
    K: float,
    notional: float,
    T: float,
    r: float,
    b: float,
    sigma: float,
) -> float:
    """资产或无看涨期权。"""
    d = _d1(S, K, T, b, sigma)
    return notional * np.exp(-r * T) * norm.cdf(d)


def asset_or_nothing_put(
    S: float,
    K: float,
    notional: float,
    T: float,
    r: float,
    b: float,
    sigma: float,
) -> float:
    """资产或无看跌期权。"""
    d = _d1(S, K, T, b, sigma)
    return notional * np.exp(-r * T) * norm.cdf(-d)


def supershare_option_price(
    S: float,
    K_low: float,
    K_high: float,
    T: float,
    r: float,
    b: float,
    sigma: float,
) -> tuple[float, float, float]:
    """Supershare 期权定价，返回 (price1, price2, weight)。"""
    d1 = _d1(S, K_low, T, b, sigma)
    d2 = _d1(S, K_high, T, b, sigma)
    price1 = np.exp(-r * T) * norm.cdf(d1)
    price2 = np.exp(-r * T) * norm.cdf(d2)
    weight = (S * np.exp((b - r) * T) / K_low) * (norm.cdf(d1) - norm.cdf(d2))
    return price1, price2, weight
