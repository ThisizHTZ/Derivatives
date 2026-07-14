"""BSM 与二叉树定价示例。"""

from derivatives.models.binomial_tree import european_call_tree, european_put_tree
from derivatives.pricing.bsm import black_scholes_merton


def main() -> None:
    S, K, T, r, sigma = 100.0, 120.0, 0.5, 0.05, 0.2
    q = 0.02

    call = black_scholes_merton(S, K, T, r, sigma, "call")
    put = black_scholes_merton(S, K, T, r, sigma, "put")
    print(f"BSM Call: {call:.3f}")
    print(f"BSM Put:  {put:.3f}")

    tree_call = european_call_tree(S, K, r, sigma, T, steps=100)
    tree_put = european_put_tree(S, K, r, q, sigma, T, steps=1000)
    print(f"Tree Call: {tree_call:.3f}")
    print(f"Tree Put:  {tree_put:.2f}")


if __name__ == "__main__":
    main()
