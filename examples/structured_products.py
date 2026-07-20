"""结构化产品定价示例。"""

from derivatives.pricing.structured import (
    airbag_option_price,
    range_accrual_price,
    shark_fin_option,
    up_and_out_call,
)
from derivatives.pricing.exotic import (
    arithmetic_asian_option_price,
    basket_option_price,
    capital_protected_note_price,
    fixed_strike_lookback_option_price,
    phoenix_autocall_note_price,
    snowball_note_price,
)


def main() -> None:
    shark_call = shark_fin_option(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
        upper_barrier=120,
        steps=100,
        simulations=10_000,
        option_type="call",
        seed=42,
    )
    print(f"鲨鱼鳍看涨: {shark_call:.4f}")

    shark_fin = up_and_out_call(
        S=100,
        X=92,
        T=91 / 365,
        r=0.02,
        q=0.0,
        sigma=0.07,
        H=109,
        K=2.45,
        participate_rate=0.2647,
        floor_return=0.5,
    )
    print(f"向上敲出结构: {shark_fin:.4f}")

    airbag = airbag_option_price(
        S=100,
        K=100,
        H=70,
        T=1,
        r=0.05,
        sigma=0.2,
        participate_rate=0.7,
    )
    print(f"安全气囊: {airbag:.5f}")

    range_price = range_accrual_price(
        S0=100,
        T=1.0,
        sigma=0.2,
        r=0.05,
        lower_bound=90,
        upper_bound=120,
        trading_days=240,
        paths=1000,
        seed=42,
    )
    print(f"Range Accrual: {range_price:.4f}")

    protected_note = capital_protected_note_price(
        S=100,
        K=100,
        T=1,
        r=0.03,
        sigma=0.2,
        notional=100,
        participation_rate=0.8,
    )
    print(f"保本看涨票据（含本金现值）: {protected_note:.4f}")

    asian = arithmetic_asian_option_price(
        S0=100,
        K=100,
        T=1,
        r=0.03,
        sigma=0.2,
        steps=252,
        paths=20_000,
        seed=42,
    )
    lookback = fixed_strike_lookback_option_price(
        S0=100,
        K=100,
        T=1,
        r=0.03,
        sigma=0.2,
        steps=252,
        paths=20_000,
        seed=42,
    )
    print(f"算术平均亚式看涨: {asian:.4f}")
    print(f"固定行权价回望看涨: {lookback:.4f}")

    basket = basket_option_price(
        spots=[100, 100],
        weights=[0.5, 0.5],
        K=100,
        T=1,
        r=0.03,
        volatilities=[0.2, 0.25],
        correlation=[[1.0, 0.4], [0.4, 1.0]],
        paths=20_000,
        seed=42,
    )
    print(f"双资产算术篮子看涨: {basket:.4f}")

    observation_times = [i / 12 for i in range(1, 13)]
    snowball = snowball_note_price(
        S0=100,
        T=1,
        r=0.03,
        sigma=0.2,
        observation_times=observation_times,
        knock_out_barriers=[1.03 - 0.005 * i for i in range(12)],
        knock_in_barrier=0.7,
        coupon_rate=0.12,
        steps=252,
        paths=20_000,
        seed=42,
    )
    print(
        f"雪球现值: {snowball.price:.4f}, "
        f"敲出概率: {snowball.autocall_probability:.1%}, "
        f"敲入概率: {snowball.knock_in_probability:.1%}, "
        f"预期存续期: {snowball.expected_life:.2f} 年"
    )

    phoenix = phoenix_autocall_note_price(
        S0=100,
        T=1,
        r=0.03,
        sigma=0.2,
        observation_times=observation_times,
        autocall_barriers=[1.0] * 12,
        coupon_barrier=0.75,
        coupon_rate=0.10,
        protection_barrier=0.65,
        memory_coupon=True,
        steps=252,
        paths=20_000,
        seed=42,
    )
    print(
        f"Phoenix 现值: {phoenix.price:.4f}, "
        f"自动赎回概率: {phoenix.autocall_probability:.1%}, "
        f"本金损失概率: {phoenix.loss_probability:.1%}"
    )


if __name__ == "__main__":
    main()
