"""障碍与敲入期权示例。"""

from derivatives.pricing.barrier import double_barrier_call, double_barrier_put
from derivatives.pricing.knock_in import down_and_in_call, up_and_in_call


def main() -> None:
    S, K, T, r, sigma = 100.0, 90.0, 1.0, 0.05, 0.2
    upper, lower = 120.0, 80.0

    print(f"双障碍看涨: {double_barrier_call(S, K, T, r, sigma, upper, lower):.4f}")
    print(f"双障碍看跌: {double_barrier_put(S, K, T, r, sigma, upper, lower):.4f}")

    H = 90.0
    rebate = 0.02
    di = down_and_in_call(S=105, X=95, T=1, r=0.04, sigma=0.5, H=H, K=rebate)
    ui = up_and_in_call(S=105, X=95, T=1, r=0.04, sigma=0.5, H=H, K=rebate)
    print(f"向下敲入看涨: {di}")
    print(f"向上敲入看涨: {ui}")


if __name__ == "__main__":
    main()
