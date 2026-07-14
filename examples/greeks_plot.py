"""Greeks 计算与可视化示例。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from derivatives.analytics.greeks import compute_greeks

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    S_grid = np.linspace(0.1, 200, 100)
    result = compute_greeks("C", S_grid, K=100, sigma=0.2, T=1.0, r=0.05, b=0.05)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
    greek_pairs = [["option_value", "delta"], ["gamma", "vega"], ["theta", "rho"]]
    for row, pair in enumerate(greek_pairs):
        for col, name in enumerate(pair):
            axes[row, col].plot(S_grid, result[name])
            axes[row, col].set_title(name)
            axes[row, col].legend([name])

    output = Path("artifacts/greeks.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(f"Greeks 图表已保存至 {output}")


if __name__ == "__main__":
    main()
