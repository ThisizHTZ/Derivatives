"""路径依赖与结构化产品测试。"""

import numpy as np
import pytest

from derivatives.pricing.bsm import black_scholes_merton
from derivatives.pricing.exotic import (
    arithmetic_asian_option_price,
    basket_option_price,
    capital_protected_note_price,
    fixed_strike_lookback_option_price,
    phoenix_autocall_note_price,
    snowball_note_price,
)


def test_capital_protected_note_has_principal_floor_value():
    price = capital_protected_note_price(
        S=100,
        K=100,
        T=1,
        r=0.03,
        sigma=0.2,
        notional=100,
        participation_rate=0.8,
    )
    assert price > 100 * np.exp(-0.03)


def test_asian_call_is_reproducible_and_below_vanilla():
    kwargs = dict(S0=100, K=100, T=1, r=0.03, sigma=0.2, steps=64, paths=20_000, seed=42)
    asian_1 = arithmetic_asian_option_price(**kwargs)
    asian_2 = arithmetic_asian_option_price(**kwargs)
    vanilla = black_scholes_merton(100, 100, 1, 0.03, 0.2, "call")
    assert asian_1 == asian_2
    assert 0 < asian_1 < vanilla


def test_lookback_call_is_more_valuable_than_vanilla():
    lookback = fixed_strike_lookback_option_price(
        S0=100,
        K=100,
        T=1,
        r=0.03,
        sigma=0.2,
        steps=64,
        paths=20_000,
        seed=42,
    )
    vanilla = black_scholes_merton(100, 100, 1, 0.03, 0.2, "call")
    assert lookback > vanilla


def test_single_asset_basket_matches_vanilla():
    basket = basket_option_price(
        spots=[100],
        weights=[1],
        K=100,
        T=1,
        r=0.03,
        volatilities=[0.2],
        correlation=[[1.0]],
        paths=100_000,
        seed=42,
    )
    vanilla = black_scholes_merton(100, 100, 1, 0.03, 0.2, "call")
    assert basket == pytest.approx(vanilla, rel=0.03)


def test_basket_rejects_invalid_correlation():
    with pytest.raises(ValueError, match="positive semidefinite"):
        basket_option_price(
            spots=[100, 100, 100],
            weights=[1 / 3] * 3,
            K=100,
            T=1,
            r=0.03,
            volatilities=[0.2] * 3,
            correlation=[[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]],
        )


def test_snowball_autocalls_at_first_observation_in_deterministic_case():
    result = snowball_note_price(
        S0=100,
        T=1,
        r=0,
        sigma=0,
        observation_times=[0.25, 0.5, 0.75, 1.0],
        knock_out_barriers=[1.0] * 4,
        knock_in_barrier=0.7,
        coupon_rate=0.08,
        steps=4,
        paths=100,
        seed=42,
    )
    assert result.price == pytest.approx(102.0)
    assert result.autocall_probability == 1.0
    assert result.expected_life == pytest.approx(0.25)
    assert result.knock_in_probability == 0.0


def test_snowball_bears_loss_after_knock_in():
    result = snowball_note_price(
        S0=100,
        T=1,
        r=0,
        q=0.5,
        sigma=0,
        observation_times=[0.5, 1.0],
        knock_out_barriers=[2.0, 2.0],
        knock_in_barrier=0.9,
        coupon_rate=0.08,
        steps=4,
        paths=100,
        seed=42,
    )
    assert result.price == pytest.approx(100 * np.exp(-0.5))
    assert result.knock_in_probability == 1.0
    assert result.loss_probability == 1.0


def test_phoenix_autocalls_and_pays_period_coupon():
    result = phoenix_autocall_note_price(
        S0=100,
        T=1,
        r=0,
        sigma=0,
        observation_times=[0.25, 0.5, 0.75, 1.0],
        autocall_barriers=[1.0] * 4,
        coupon_barrier=0.8,
        coupon_rate=0.08,
        protection_barrier=0.7,
        steps=4,
        paths=100,
        seed=42,
    )
    assert result.price == pytest.approx(102.0)
    assert result.autocall_probability == 1.0
    assert result.loss_probability == 0.0
