"""基差率与股息率相关计算。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_basis_rates(file_path: str | Path) -> pd.DataFrame:
    """从中证500数据文件计算年化基差率与二十日移动均值。

    Parameters
    ----------
    file_path : str or Path
        Excel 数据文件路径，需包含日期、指数价格、期货价格等列。

    Returns
    -------
    pd.DataFrame
        含 ``annualized_basis_rate`` 与 ``custom_moving_avg_basis_rate`` 的结果表。
    """
    df = pd.read_excel(file_path)
    df.columns = [
        "date",
        "index_price",
        "future_price",
        "near_month_position",
        "next_near_month_price",
        "next_near_month_position",
    ]
    df["date"] = pd.to_datetime(df["date"])
    df["future_expiry_date"] = df["date"] + pd.offsets.MonthEnd(0)

    def annualized_basis_rate(row: pd.Series) -> float:
        days_to_expiry = (row["future_expiry_date"] - row["date"]).days
        if days_to_expiry == 0:
            return np.nan
        return (365 / days_to_expiry) * np.log(row["index_price"] / row["future_price"])

    df["annualized_basis_rate"] = df.apply(annualized_basis_rate, axis=1)

    def moving_avg_basis_rate(t: int) -> float:
        if t <= 20:
            return np.nan
        rates = [
            (np.log(df.loc[idx, "index_price"] / df.loc[idx, "future_price"]) * 365)
            / (df.loc[idx, "future_expiry_date"] - df.loc[idx, "date"]).days
            for idx in range(t - 20, t)
            if (df.loc[idx, "future_expiry_date"] - df.loc[idx, "date"]).days != 0
        ]
        return float(np.mean(rates)) if rates else np.nan

    df["custom_moving_avg_basis_rate"] = df.index.map(moving_avg_basis_rate)
    return df[
        [
            "date",
            "index_price",
            "future_price",
            "annualized_basis_rate",
            "custom_moving_avg_basis_rate",
        ]
    ]
