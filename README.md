# Derivatives Toolkit

金融衍生品定价与研究工具包，覆盖 BSM 解析解、二叉树、蒙特卡洛、障碍/敲入期权、结构化产品（鲨鱼鳍、安全气囊、Range Accrual）以及 Greeks 与基差率分析。

## 项目结构

```
derivatives/                 # 核心 Python 包
  pricing/                   # 定价模型
    bsm.py                   # Black-Scholes-Merton
    binary.py                # 二元/现金或无/资产或无
    barrier.py               # 单/双障碍期权
    knock_in.py              # 敲入期权
    structured.py            # 鲨鱼鳍、安全气囊、Range Accrual
  models/                    # 数值方法
    binomial_tree.py         # 欧式/美式二叉树
    monte_carlo.py           # 蒙特卡洛
    brownian.py              # 布朗运动
  analytics/                 # 分析工具
    greeks.py                # 希腊字母
    basis_rate.py            # 基差率计算
  integrations/
    windlink.py              # Wind 接口（可选）

examples/                    # 可运行示例
tests/                       # 单元测试
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
# 或开发模式安装
pip install -e ".[dev]"
```

### 代码调用

```python
from derivatives import black_scholes_merton, compute_greeks, shark_fin_option

call_price = black_scholes_merton(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
greeks = compute_greeks("C", S=100, K=100, sigma=0.2, T=1, r=0.05, b=0.05)
shark = shark_fin_option(S=100, K=100, T=1, r=0.05, sigma=0.2, upper_barrier=120, steps=100, simulations=10000, seed=42)
```

### 运行示例

```bash
python -m examples.bsm_and_tree
python -m examples.barrier_and_knock_in
python -m examples.structured_products
python -m examples.monte_carlo
python -m examples.greeks_plot
python -m examples.basis_rate
```

### 运行测试

```bash
pytest -q
```

## 模块索引

| 模块 | 功能 |
|------|------|
| `derivatives.pricing.bsm` | 欧式看涨/看跌 BSM 定价 |
| `derivatives.pricing.binary` | 二元、现金或无、资产或无、Supershare |
| `derivatives.pricing.barrier` | 双障碍、含股息双向障碍、二元障碍 |
| `derivatives.pricing.knock_in` | 向上/向下敲入看涨 |
| `derivatives.pricing.structured` | 鲨鱼鳍、安全气囊、Range Accrual |
| `derivatives.models.binomial_tree` | 欧式/美式二叉树 |
| `derivatives.models.monte_carlo` | GBM 路径与 MC 定价 |
| `derivatives.analytics.greeks` | Delta/Gamma/Vega/Theta/Rho |
| `derivatives.analytics.basis_rate` | 中证500基差率与20日均值 |

## 数据文件

| 文件 | 说明 |
|------|------|
| `中证500数据.xlsx` | 基差率计算示例数据 |
| `500历史数据更新.xlsx` | 历史行情数据 |
| `updated_custom_moving_avg_basis_rates4.xlsx` | 基差率计算结果 |

## 旧脚本迁移

根目录下的中文命名脚本已改为兼容入口，内部转发至 `examples/` 或 `derivatives` 包。推荐直接使用包 API 或 `examples/` 模块。

| 旧文件 | 新入口 |
|--------|--------|
| `main.py` | `examples/structured_products.py` + `examples/barrier_and_knock_in.py` |
| `BSM和Binary_tree.py` | `examples/bsm_and_tree.py` |
| `Greeks希腊函数.py` | `examples/greeks_plot.py` |
| `蒙特卡洛欧式期权.py` | `examples/monte_carlo.py` |
| `分红率或股息率.py` | `examples/basis_rate.py` |

## 工程改进说明（v0.2.0）

- 抽取 `derivatives` 包，消除 `main.py` 中的重复代码
- 修复 `Greeks` 看涨分支返回值错误
- 统一函数命名（`S/K` 参数、`snake_case`）
- 所有示例脚本增加 `if __name__ == "__main__"` 守卫
- 蒙特卡洛支持 `seed` 参数，结果可复现
- 添加 `requirements.txt`、`pyproject.toml`、`.gitignore`、单元测试

## 许可证

见 [LICENSE](LICENSE)。
