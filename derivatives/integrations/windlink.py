"""Wind 数据接口封装（可选依赖）。"""

from __future__ import annotations


def start_wind() -> object | None:
    """启动 Wind 连接。

    Returns
    -------
    object or None
        Wind 实例；若未安装 WindPy 则返回 None。
    """
    try:
        from WindPy import w  # type: ignore
    except ImportError:
        return None

    w.start()
    return w
