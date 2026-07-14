"""欧式期权希腊字母计算。"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy import exp
from scipy.stats import norm


class GreeksResult(TypedDict):
    option_value: float | np.ndarray
    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    theta: float | np.ndarray
    rho: float | np.ndarray


def compute_greeks(
    option_type: str,
    S: float | np.ndarray,
    K: float,
    sigma: float,
    T: float,
    r: float,
    b: float,
) -> GreeksResult:
    """计算欧式期权的估值与希腊字母。

    Parameters
    ----------
    option_type : str
        ``'C'`` 或 ``'P'``。
    S : float or ndarray
        标的价格，可传入序列用于批量计算。
    K : float
        行权价。
    sigma : float
        波动率。
    T : float
        到期时间（年）。
    r : float
        无风险利率。
    b : float
        持有成本；``b=r`` 无股利，``b=0`` 期货，``b=r-q`` 支付股利。
    """
    d1 = (np.log(S / K) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.upper() == "C":
        option_value = S * exp((b - r) * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = exp((b - r) * T) * norm.cdf(d1)
        theta = (
            -exp((b - r) * T) * S * norm.pdf(d1) * sigma / (2 * T**0.5)
            - r * K * exp(-r * T) * norm.cdf(d2)
            - (b - r) * S * exp((b - r) * T) * norm.cdf(d1)
        )
        if b != 0:
            rho = K * T * exp(-r * T) * norm.cdf(d2)
        else:
            rho = -T * exp(-r * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        option_value = K * exp(-r * T) * norm.cdf(-d2) - S * exp((b - r) * T) * norm.cdf(-d1)
        delta = -exp((b - r) * T) * norm.cdf(-d1)
        theta = (
            -exp((b - r) * T) * S * norm.pdf(d1) * sigma / (2 * T**0.5)
            + r * K * exp(-r * T) * norm.cdf(-d2)
            + (b - r) * S * exp((b - r) * T) * norm.cdf(-d1)
        )
        if b != 0:
            rho = -K * T * exp(-r * T) * norm.cdf(-d2)
        else:
            rho = -T * exp(-r * T) * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

    gamma = exp((b - r) * T) * norm.pdf(d1) / (S * sigma * T**0.5)
    vega = S * exp((b - r) * T) * norm.pdf(d1) * T**0.5

    return {
        "option_value": option_value,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
