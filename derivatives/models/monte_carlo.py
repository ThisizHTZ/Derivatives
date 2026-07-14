"""蒙特卡洛模拟与几何布朗运动。"""

from __future__ import annotations

import numpy as np


def geo_brownian_motion(
    steps: int,
    paths: int,
    T: float,
    S0: float,
    b: float,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """生成几何布朗运动价格路径。

    Returns
    -------
    np.ndarray
        形状为 ``(steps + 1, paths)`` 的价格矩阵。
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    path = np.zeros((steps + 1, paths))
    path[0] = S0

    for step in range(1, steps + 1):
        shocks = rng.standard_normal(paths)
        path[step] = path[step - 1] * np.exp((b - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks)

    return path


def european_call_mc(
    steps: int,
    paths: int,
    T: float,
    S0: float,
    K: float,
    sigma: float,
    r: float,
    b: float,
    seed: int | None = None,
) -> float:
    """欧式看涨期权蒙特卡洛定价。

    Notes
    -----
    ``b=r`` 为无股利期权，``b=r-q`` 为支付股利期权，``b=0`` 为期货期权。
    """
    price_paths = geo_brownian_motion(steps, paths, T, S0, b, sigma, seed=seed)
    payoff = np.maximum(price_paths[-1] - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())
