"""布朗运动路径模拟。"""

from __future__ import annotations

import numpy as np


def standard_brownian_motion(
    steps: int,
    paths: int,
    T: float,
    S0: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """标准布朗运动路径模拟。

    Returns
    -------
    np.ndarray
        形状为 ``(steps + 1, paths)`` 的路径矩阵。
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    path = np.zeros((steps + 1, paths))
    path[0] = S0
    shocks = rng.standard_normal((steps, paths))

    for step in range(1, steps + 1):
        path[step] = path[step - 1] + shocks[step - 1] * np.sqrt(dt)

    return path
