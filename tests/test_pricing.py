"""定价模块单元测试。"""

import numpy as np
import pytest

from derivatives.analytics.greeks import compute_greeks
from derivatives.models.binomial_tree import european_call_tree
from derivatives.models.monte_carlo import european_call_mc
from derivatives.pricing.barrier import double_barrier_call
from derivatives.pricing.bsm import black_scholes_merton
from derivatives.pricing.knock_in import down_and_in_call
from derivatives.pricing.structured import airbag_option_price, shark_fin_option


@pytest.fixture
def market_params():
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


def test_black_scholes_put_call_parity(market_params):
    p = market_params
    call = black_scholes_merton(**p, option_type="call")
    put = black_scholes_merton(**p, option_type="put")
    parity = call - put - p["S"] + p["K"] * np.exp(-p["r"] * p["T"])
    assert parity == pytest.approx(0.0, abs=1e-10)


def test_tree_converges_to_bsm(market_params):
    p = market_params
    analytical = black_scholes_merton(**p, option_type="call")
    tree = european_call_tree(p["S"], p["K"], p["r"], p["sigma"], p["T"], steps=500)
    assert tree == pytest.approx(analytical, rel=0.02)


def test_monte_carlo_near_bsm(market_params):
    p = market_params
    analytical = black_scholes_merton(**p, option_type="call")
    mc = european_call_mc(
        steps=252,
        paths=100_000,
        T=p["T"],
        S0=p["S"],
        K=p["K"],
        sigma=p["sigma"],
        r=p["r"],
        b=p["r"],
        seed=42,
    )
    assert mc == pytest.approx(analytical, rel=0.05)


def test_greeks_call_returns_all_keys(market_params):
    p = market_params
    result = compute_greeks("C", p["S"], p["K"], p["sigma"], p["T"], p["r"], p["r"])
    assert set(result.keys()) == {"option_value", "delta", "gamma", "vega", "theta", "rho"}
    assert result["delta"] > 0


def test_greeks_put_delta_negative(market_params):
    p = market_params
    result = compute_greeks("P", p["S"], p["K"], p["sigma"], p["T"], p["r"], p["r"])
    assert result["delta"] < 0


def test_double_barrier_positive():
    price = double_barrier_call(100, 90, 1, 0.05, 0.2, 120, 80)
    assert price >= 0


def test_knock_in_returns_tuple():
    result = down_and_in_call(105, 95, 1, 0.04, 0.5, 90, 0.02)
    assert len(result) == 2


def test_airbag_positive():
    price = airbag_option_price(100, 100, 70, 1, 0.05, 0.2, 0.7)
    assert price > 0


def test_shark_fin_with_seed_reproducible():
    kwargs = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, upper_barrier=120, steps=50, simulations=5000)
    p1 = shark_fin_option(**kwargs, seed=7)
    p2 = shark_fin_option(**kwargs, seed=7)
    assert p1 == p2
