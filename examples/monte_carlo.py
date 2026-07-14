"""蒙特卡洛与布朗运动示例。"""

from derivatives.models.brownian import standard_brownian_motion
from derivatives.models.monte_carlo import european_call_mc


def main() -> None:
    mc_price = european_call_mc(
        steps=250,
        paths=50_000,
        T=1,
        S0=100,
        K=99,
        sigma=0.2,
        r=0.03,
        b=0.03,
        seed=42,
    )
    print(f"蒙特卡洛欧式看涨: {mc_price:.4f}")

    path = standard_brownian_motion(steps=100, paths=10, T=1, S0=0, seed=42)
    print(f"布朗运动路径形状: {path.shape}")


if __name__ == "__main__":
    main()
