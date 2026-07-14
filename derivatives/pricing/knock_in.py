"""敲入类障碍期权定价。"""

from __future__ import annotations

import math

from scipy.stats import norm


def down_and_in_call(
    S: float,
    X: float,
    T: float,
    r: float,
    sigma: float,
    H: float,
    K: float,
) -> tuple[float, float]:
    """向下敲入看涨期权。

    Returns
    -------
    tuple[float, float]
        (X >= H 情形价格, X < H 情形价格)
    """
    mu = (r - 0.5 * sigma**2) / sigma**2
    lambda_ = math.sqrt(mu**2 + 2 * r / sigma**2)

    x1 = (math.log(S / X) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    x2 = (math.log(S / H) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y1 = (math.log(H**2 / (S * X)) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y2 = (math.log(H / S) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    z = (math.log(H / S) + lambda_ * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))

    a = S * math.exp((r - sigma**2 / 2) * T) * norm.cdf(x1) - X * math.exp(-r * T) * norm.cdf(
        x1 - sigma * math.sqrt(T)
    )
    b = S * math.exp((r - sigma**2 / 2) * T) * norm.cdf(x2) - X * math.exp(-r * T) * norm.cdf(
        x2 - sigma * math.sqrt(T)
    )
    c = S * math.exp((r - sigma**2 / 2) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(y1) - X * math.exp(
        -r * T
    ) * (H / S) ** (2 * mu) * norm.cdf(y1 - sigma * math.sqrt(T))
    d = S * math.exp((r - sigma**2 / 2) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(y2) - X * math.exp(
        -r * T
    ) * (H / S) ** (2 * mu) * norm.cdf(y2 - sigma * math.sqrt(T))
    e = K * math.exp(-r * T) * (norm.cdf(-z) - (H / S) ** (2 * mu) * norm.cdf(y2 - sigma * math.sqrt(T)))

    price_x_ge_h = c + e
    price_x_lt_h = a - b + d + e
    return price_x_ge_h, price_x_lt_h


def up_and_in_call(
    S: float,
    X: float,
    T: float,
    r: float,
    sigma: float,
    H: float,
    K: float,
) -> tuple[float, float]:
    """向上敲入看涨期权。

    Returns
    -------
    tuple[float, float]
        (X >= H 情形价格, X < H 情形价格)
    """
    mu = (r - 0.5 * sigma**2) / sigma**2
    lambda_ = math.sqrt(-mu**2 + 2 * r / sigma**2)

    x1 = (math.log(S / X) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    x2 = (math.log(S / H) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y1 = (math.log(H**2 / (S * X)) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y2 = (math.log(H / S) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    z = (math.log(H / S) + lambda_ * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))

    a = S * math.exp((r - sigma**2 / 2) * T) * norm.cdf(x1) - X * math.exp(-r * T) * norm.cdf(
        x1 - sigma * math.sqrt(T)
    )
    b = S * math.exp((r - sigma**2 / 2) * T) * norm.cdf(x2) - X * math.exp(-r * T) * norm.cdf(
        x2 - sigma * math.sqrt(T)
    )
    c = S * math.exp((r - sigma**2 / 2) * T) * (H / S) ** (2 * (-mu + 1)) * norm.cdf(y1) - X * math.exp(
        -r * T
    ) * (H / S) ** (2 * -mu) * norm.cdf(y1 - sigma * math.sqrt(T))
    d = S * math.exp((r - sigma**2 / 2) * T) * (H / S) ** (2 * (-mu + 1)) * norm.cdf(y2) - X * math.exp(
        -r * T
    ) * (H / S) ** (2 * -mu) * norm.cdf(y2 - sigma * math.sqrt(T))
    e = K * math.exp(-r * T) * (norm.cdf(-z) - (H / S) ** (2 * -mu) * norm.cdf(y2 - sigma * math.sqrt(T)))

    price_x_ge_h = a + e
    price_x_lt_h = b - c + d + e
    return price_x_ge_h, price_x_lt_h
