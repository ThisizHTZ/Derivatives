"""基差率计算示例。"""

from pathlib import Path

from derivatives.analytics.basis_rate import compute_basis_rates


def main() -> None:
    data_file = Path("中证500数据.xlsx")
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return

    result = compute_basis_rates(data_file)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
