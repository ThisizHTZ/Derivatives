"""兼容入口。"""

from derivatives.integrations.windlink import start_wind

if __name__ == "__main__":
    client = start_wind()
    if client is None:
        print("WindPy 未安装，请在本机 Wind 终端环境中安装后使用。")
    else:
        print("Wind 连接已启动。")
