"""Black-Scholes-Merton 欧式期权定价。"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_merton(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """计算欧式期权理论价格。

    Parameters
    ----------
    S : float
        标的资产当前价格。
    K : float
        行权价格。
    T : float
        到期时间（年）。
    r : float
        无风险利率（年化）。
    sigma : float
        波动率（年化）。
    option_type : str
        ``'call'`` 或 ``'put'``。
    q : float
        连续股息率（年化），默认 0。

    Returns
    -------
    float
        期权理论价格。
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    b = r - q
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")
