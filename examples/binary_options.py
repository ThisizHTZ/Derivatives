"""二元期权定价示例。"""

from derivatives.pricing.binary import (
    asset_or_nothing_call,
    asset_or_nothing_put,
    binary_option_price,
    cash_or_nothing_call,
    cash_or_nothing_put,
    supershare_option_price,
)


def main() -> None:
    print(f"二元看涨: {binary_option_price(100, 90, 1, 0.05, 0.2, 'call'):.4f}")

    cash_call = cash_or_nothing_call(100, 80, 10, 0.75, 0.06, 0.0, 0.35)
    cash_put = cash_or_nothing_put(100, 80, 10, 0.75, 0.06, 0.0, 0.35)
    print(f"现金或无看涨: {cash_call:.5f}")
    print(f"现金或无看跌: {cash_put:.5f}")

    asset_call = asset_or_nothing_call(70, 65, 10, 0.5, 0.07, 0.02, 0.27)
    asset_put = asset_or_nothing_put(70, 65, 10, 0.5, 0.07, 0.02, 0.27)
    print(f"资产或无看涨: {asset_call:.5f}")
    print(f"资产或无看跌: {asset_put:.5f}")

    supershare = supershare_option_price(100, 90, 110, 0.25, 0.1, 0.0, 0.2)
    print(f"Supershare: {supershare}")


if __name__ == "__main__":
    main()
