# 01 · 快速上手

## 1. 环境

```bash
# 建议 Python 3.10+
pip install -r requirements.txt
# 开发模式（推荐，改代码即可生效）
pip install -e ".[dev]"
```

验证：

```bash
python3 -m pytest -q
python3 -m examples.bsm_and_tree
```

## 2. 最小可运行例子

### 2.1 香草欧式

```python
from derivatives import black_scholes_merton

call = black_scholes_merton(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
put = black_scholes_merton(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
print(call, put)
```

自检：`call - put ≈ S - K*exp(-rT)`（无股息时）。

### 2.2 Greeks

```python
from derivatives import compute_greeks

g = compute_greeks("C", S=100, K=100, sigma=0.2, T=1, r=0.05, b=0.05)
print(g["delta"], g["vega"], g["theta"])
```

注意：本库 Vega 是对波动率的偏导（未按「每 1%」缩放）。交易台习惯常报 **vega / 100**（每 1 vol point）。

### 2.3 鲨鱼鳍（蒙特卡洛）

```python
from derivatives import shark_fin_option

px = shark_fin_option(
    S=100, K=100, T=1, r=0.05, sigma=0.2,
    upper_barrier=120, steps=100, simulations=20_000,
    option_type="call", seed=42,
)
print(px)
```

务必传 `seed`，否则每次结果不同，无法复现实验。

## 3. 示例脚本一览

| 命令 | 内容 |
|------|------|
| `python3 -m examples.bsm_and_tree` | BSM + 二叉树对照 |
| `python3 -m examples.binary_options` | 二元 / 现金或无 / Supershare |
| `python3 -m examples.barrier_and_knock_in` | 双障碍与敲入 |
| `python3 -m examples.structured_products` | 鲨鱼鳍、气囊、Range Accrual |
| `python3 -m examples.monte_carlo` | MC 欧式 + 布朗运动 |
| `python3 -m examples.greeks_plot` | Greeks 随 S 变化图 |
| `python3 -m examples.basis_rate` | 中证500基差率（需本地 Excel） |

## 4. 旧脚本怎么用

根目录中文文件名脚本仍可运行，内部已转发到 `examples/`。  
长期建议：**直接 import `derivatives`**，不要再复制粘贴根目录逻辑。

## 5. 推荐练习（比作业更接近实务）

1. **对照练习**：同一组 `(S,K,T,r,sigma)`，分别用 BSM、二叉树(steps=100/500)、MC(paths=1e5) 定价，画误差随步数变化。  
2. **分红敏感性**：固定其他参数，扫 `q` 从 0 到 5%，看 call/put 如何变化。  
3. **障碍监控**：同一障碍，把 MC 的 `steps` 从 50 提到 252，观察价格变化（离散监控偏差）。  
4. **基差作业**：用 `中证500数据.xlsx` 算年化基差率，讨论「负基差」时对指数期权定价的含义。  
5. **条款翻译**：找一份公开的银行结构性存款说明书，把收益公式画成 payoff 图，再映射到本库函数。

下一章：[02-定价实践：课本之外](02-pricing-beyond-textbook.md)
