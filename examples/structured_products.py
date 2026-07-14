"""结构化产品定价示例。"""

from derivatives.pricing.structured import (
    airbag_option_price,
    range_accrual_price,
    shark_fin_option,
    up_and_out_call,
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


if __name__ == "__main__":
    main()
