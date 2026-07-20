"""常见路径依赖与结构化产品的蒙特卡洛定价。

本模块按投资者视角返回含本金的现值。所有障碍均使用相对期初价格的比例，
例如 ``0.7`` 表示期初价格的 70%。模型采用常数波动率 GBM，适合教学、
条款原型与交叉验证，不应直接替代生产级波动率曲面和跳跃模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class StructuredNoteResult:
    """自动赎回类产品的估值与路径统计。

    Attributes
    ----------
    price:
        投资者未来现金流的含本金现值。
    standard_error:
        蒙特卡洛价格标准误。
    autocall_probability:
        到期前或到期观察日触发自动赎回的模拟概率。
    knock_in_probability:
        存续期内触及下方敲入障碍的模拟概率；无敲入条款时为 0。
    loss_probability:
        到期本金低于名义本金的模拟概率。
    expected_life:
        各路径兑付时间的平均值（年）。
    """

    price: float
    standard_error: float
    autocall_probability: float
    knock_in_probability: float
    loss_probability: float
    expected_life: float


def _validate_mc(T: float, sigma: float, steps: int, paths: int) -> None:
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if paths < 2:
        raise ValueError("paths must be at least 2")


def _gbm_paths(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    steps: int,
    paths: int,
    seed: int | None,
) -> np.ndarray:
    """生成包含期初价格的 GBM 路径矩阵。"""
    _validate_mc(T, sigma, steps, paths)
    if S0 <= 0:
        raise ValueError("S0 must be positive")

    rng = np.random.default_rng(seed)
    dt = T / steps
    shocks = rng.standard_normal((paths, steps))
    log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.cumsum(log_returns, axis=1)
    simulated = S0 * np.exp(log_paths)
    return np.concatenate((np.full((paths, 1), S0), simulated), axis=1)


def _option_payoff(underlying: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(underlying - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - underlying, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def _observation_indices(observation_times: Sequence[float], T: float, steps: int) -> np.ndarray:
    times = np.asarray(observation_times, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("observation_times must be a non-empty one-dimensional sequence")
    if np.any(times <= 0) or np.any(times > T) or np.any(np.diff(times) <= 0):
        raise ValueError("observation_times must be strictly increasing and within (0, T]")

    indices = np.rint(times / T * steps).astype(int)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("steps is too small to distinguish all observation times")
    return indices


def _standard_error(discounted_values: np.ndarray) -> float:
    return float(np.std(discounted_values, ddof=1) / np.sqrt(len(discounted_values)))


def capital_protected_note_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    notional: float = 100.0,
    participation_rate: float = 1.0,
    q: float = 0.0,
) -> float:
    """保本看涨型票据：零息债券 + 按名义本金缩放的欧式看涨。

    到期兑付为 ``notional * (1 + participation_rate * max(S_T/K - 1, 0))``。
    """
    from derivatives.pricing.bsm import black_scholes_merton

    if min(S, K, notional) <= 0 or participation_rate < 0:
        raise ValueError("S, K and notional must be positive; participation_rate must be non-negative")
    call = black_scholes_merton(S, K, T, r, sigma, option_type="call", q=q)
    return float(notional * np.exp(-r * T) + notional * participation_rate * call / K)


def arithmetic_asian_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 252,
    paths: int = 50_000,
    option_type: OptionType = "call",
    q: float = 0.0,
    include_initial: bool = False,
    seed: int | None = None,
) -> float:
    """固定行权价算术平均亚式期权。

    ``include_initial`` 控制平均价格是否包含期初价；真实产品应按合同观察日设置。
    """
    if K <= 0:
        raise ValueError("K must be positive")
    prices = _gbm_paths(S0, T, r, q, sigma, steps, paths, seed)
    observed = prices if include_initial else prices[:, 1:]
    average_price = np.mean(observed, axis=1)
    payoff = _option_payoff(average_price, K, option_type)
    return float(np.exp(-r * T) * np.mean(payoff))


def fixed_strike_lookback_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 252,
    paths: int = 50_000,
    option_type: OptionType = "call",
    q: float = 0.0,
    seed: int | None = None,
) -> float:
    """离散观察的固定行权价回望期权。

    看涨使用路径最高价，看跌使用路径最低价；观察越频繁，期权通常越贵。
    """
    if K <= 0:
        raise ValueError("K must be positive")
    prices = _gbm_paths(S0, T, r, q, sigma, steps, paths, seed)
    extremum = np.max(prices, axis=1) if option_type == "call" else np.min(prices, axis=1)
    payoff = _option_payoff(extremum, K, option_type)
    return float(np.exp(-r * T) * np.mean(payoff))


def basket_option_price(
    spots: Sequence[float],
    weights: Sequence[float],
    K: float,
    T: float,
    r: float,
    volatilities: Sequence[float],
    correlation: Sequence[Sequence[float]],
    paths: int = 50_000,
    option_type: OptionType = "call",
    dividends: Sequence[float] | None = None,
    seed: int | None = None,
) -> float:
    """欧式加权算术篮子期权。

    相关矩阵必须对称半正定。权重不强制和为 1，以支持名义敞口组合。
    """
    spots_array = np.asarray(spots, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    vols_array = np.asarray(volatilities, dtype=float)
    corr_array = np.asarray(correlation, dtype=float)
    dividends_array = np.zeros_like(spots_array) if dividends is None else np.asarray(dividends, dtype=float)
    size = len(spots_array)

    if size == 0 or any(len(array) != size for array in (weights_array, vols_array, dividends_array)):
        raise ValueError("spots, weights, volatilities and dividends must have the same non-zero length")
    if corr_array.shape != (size, size):
        raise ValueError("correlation must be a square matrix matching the number of assets")
    if np.any(spots_array <= 0) or np.any(vols_array < 0) or K <= 0 or T <= 0 or paths < 2:
        raise ValueError("invalid basket or Monte Carlo parameters")
    if not np.allclose(corr_array, corr_array.T) or not np.allclose(np.diag(corr_array), 1.0):
        raise ValueError("correlation must be symmetric with a unit diagonal")
    if np.min(np.linalg.eigvalsh(corr_array)) < -1e-10:
        raise ValueError("correlation must be positive semidefinite")

    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((paths, size))
    eigenvalues, eigenvectors = np.linalg.eigh(corr_array)
    root = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    correlated = independent @ root.T
    terminal = spots_array * np.exp(
        (r - dividends_array - 0.5 * vols_array**2) * T
        + vols_array * np.sqrt(T) * correlated
    )
    basket_terminal = terminal @ weights_array
    payoff = _option_payoff(basket_terminal, K, option_type)
    return float(np.exp(-r * T) * np.mean(payoff))


def snowball_note_price(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    observation_times: Sequence[float],
    knock_out_barriers: Sequence[float],
    knock_in_barrier: float,
    coupon_rate: float,
    notional: float = 100.0,
    q: float = 0.0,
    steps: int = 252,
    paths: int = 50_000,
    seed: int | None = None,
) -> StructuredNoteResult:
    """经典敲入敲出雪球的简化估值。

    - 观察日价格达到对应敲出线：提前兑付本金及按存续年限累计的单利票息；
    - 未敲出且从未敲入：到期兑付本金及完整票息；
    - 已敲入且未敲出：若期末低于期初，兑付 ``notional * S_T/S0``，否则兑付本金。

    敲入按每个模拟步观察；敲出仅按 ``observation_times`` 观察。
    """
    indices = _observation_indices(observation_times, T, steps)
    times = np.asarray(observation_times, dtype=float)
    barriers = np.asarray(knock_out_barriers, dtype=float)
    if len(barriers) != len(times) or np.any(barriers <= 0):
        raise ValueError("knock_out_barriers must be positive and match observation_times")
    if not 0 < knock_in_barrier < 1 or coupon_rate < 0 or notional <= 0:
        raise ValueError("invalid knock-in barrier, coupon rate or notional")

    prices = _gbm_paths(S0, T, r, q, sigma, steps, paths, seed)
    ratios = prices / S0
    knocked_in = np.min(ratios[:, 1:], axis=1) <= knock_in_barrier
    active = np.ones(paths, dtype=bool)
    cashflows = np.zeros(paths)
    payment_times = np.full(paths, T, dtype=float)

    for time, index, barrier in zip(times, indices, barriers):
        called = active & (ratios[:, index] >= barrier)
        cashflows[called] = notional * (1.0 + coupon_rate * time)
        payment_times[called] = time
        active[called] = False

    terminal_ratio = ratios[:, -1]
    protected = active & ~knocked_in
    recovered = active & knocked_in & (terminal_ratio >= 1.0)
    loss = active & knocked_in & (terminal_ratio < 1.0)
    cashflows[protected] = notional * (1.0 + coupon_rate * T)
    cashflows[recovered] = notional
    cashflows[loss] = notional * terminal_ratio[loss]

    discounted = cashflows * np.exp(-r * payment_times)
    return StructuredNoteResult(
        price=float(np.mean(discounted)),
        standard_error=_standard_error(discounted),
        autocall_probability=float(np.mean(~active)),
        knock_in_probability=float(np.mean(knocked_in)),
        loss_probability=float(np.mean(loss)),
        expected_life=float(np.mean(payment_times)),
    )


def phoenix_autocall_note_price(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    observation_times: Sequence[float],
    autocall_barriers: Sequence[float],
    coupon_barrier: float,
    coupon_rate: float,
    protection_barrier: float,
    notional: float = 100.0,
    q: float = 0.0,
    memory_coupon: bool = True,
    steps: int = 252,
    paths: int = 50_000,
    seed: int | None = None,
) -> StructuredNoteResult:
    """Phoenix 自动赎回票据的简化估值。

    每个观察日：高于票息障碍则支付当期票息；``memory_coupon=True`` 时补付此前
    漏付票息。高于自动赎回线则同时赎回本金。未赎回到期且期末低于保护障碍时，
    本金按标的跌幅承担损失。票息按观察区间的年化单利计算。
    """
    indices = _observation_indices(observation_times, T, steps)
    times = np.asarray(observation_times, dtype=float)
    barriers = np.asarray(autocall_barriers, dtype=float)
    if len(barriers) != len(times) or np.any(barriers <= 0):
        raise ValueError("autocall_barriers must be positive and match observation_times")
    if not 0 < coupon_barrier <= 2 or not 0 < protection_barrier < 1:
        raise ValueError("coupon_barrier and protection_barrier are invalid")
    if coupon_rate < 0 or notional <= 0:
        raise ValueError("coupon_rate must be non-negative and notional must be positive")

    prices = _gbm_paths(S0, T, r, q, sigma, steps, paths, seed)
    ratios = prices / S0
    active = np.ones(paths, dtype=bool)
    present_values = np.zeros(paths)
    payment_times = np.full(paths, T, dtype=float)
    accrued_coupon = np.zeros(paths)
    previous_time = 0.0

    for time, index, autocall_barrier in zip(times, indices, barriers):
        period = time - previous_time
        current_coupon = notional * coupon_rate * period
        accrued_coupon[active] += current_coupon
        coupon_due = active & (ratios[:, index] >= coupon_barrier)
        coupon_payment = accrued_coupon if memory_coupon else np.full(paths, current_coupon)
        present_values[coupon_due] += coupon_payment[coupon_due] * np.exp(-r * time)
        if memory_coupon:
            accrued_coupon[coupon_due] = 0.0
        else:
            accrued_coupon[active] = 0.0

        called = active & (ratios[:, index] >= autocall_barrier)
        present_values[called] += notional * np.exp(-r * time)
        payment_times[called] = time
        active[called] = False
        previous_time = time

    terminal_ratio = ratios[:, -1]
    loss = active & (terminal_ratio < protection_barrier)
    protected_redemption = np.where(
        terminal_ratio >= protection_barrier,
        notional,
        notional * terminal_ratio,
    )
    present_values[active] += protected_redemption[active] * np.exp(-r * T)

    # 每条路径的 PV 已包含在 present_values 中，标准误可直接计算。
    return StructuredNoteResult(
        price=float(np.mean(present_values)),
        standard_error=_standard_error(present_values),
        autocall_probability=float(np.mean(~active)),
        knock_in_probability=0.0,
        loss_probability=float(np.mean(loss)),
        expected_life=float(np.mean(payment_times)),
    )
