# 09 · 新增结构化产品代码指南

新增实现位于 `derivatives/pricing/exotic.py`。共同假设：

- 标的服从常数波动率几何布朗运动（GBM）
- 利率、股息率、波动率在存续期内不变
- 自动赎回与障碍均按离散模拟步观察
- 自动赎回类函数返回**投资者视角、含本金的现值**
- 障碍使用期初价比例：`0.7` 表示期初价格的 70%

这些实现适合教学、条款原型和交叉验证；真实报价还需波动率曲面、交易日历、
信用贴现、交易费用和模型校准。

## 1. 产品与 API 对照

| 产品 | 函数 | 方法 | 主要路径依赖 |
|------|------|------|--------------|
| 保本看涨票据 | `capital_protected_note_price` | 债券 + BSM call | 无 |
| 算术平均亚式 | `arithmetic_asian_option_price` | MC | 全部观察价的平均值 |
| 固定行权价回望 | `fixed_strike_lookback_option_price` | MC | 路径最高/最低价 |
| 多资产篮子 | `basket_option_price` | 相关正态 MC | 多资产相关性 |
| 敲入敲出雪球 | `snowball_note_price` | MC | 日频敲入 + 定期敲出 |
| Phoenix 自动赎回 | `phoenix_autocall_note_price` | MC | 条件票息 + 自动赎回 |

## 2. 保本看涨票据

到期兑付：

```text
N × [1 + participation × max(S_T / K - 1, 0)]
```

因此可拆为：

```text
面值为 N 的零息债券 + (N × participation / K) 份欧式看涨
```

```python
from derivatives import capital_protected_note_price

price = capital_protected_note_price(
    S=100, K=100, T=1, r=0.03, sigma=0.2,
    notional=100, participation_rate=0.8,
)
```

注意：「保本」指发行人不违约时的合同兑付，不等于没有发行人信用风险。

## 3. 算术平均亚式期权

固定行权价看涨的 payoff：

```text
max(arithmetic_mean(S_t1, ..., S_tn) - K, 0)
```

平均价降低单日操纵与到期价格跳变的影响，通常也降低期权价值和 Vega。

```python
from derivatives import arithmetic_asian_option_price

price = arithmetic_asian_option_price(
    S0=100, K=100, T=1, r=0.03, sigma=0.2,
    steps=252, paths=50_000, include_initial=False, seed=42,
)
```

`steps` 在当前实现中同时代表模拟频率和平均价观察频率。若合同仅每月观察，
应扩展为显式 `observation_times`，不要直接把日频均价当月频均价。

## 4. 回望期权

- 固定行权价看涨：`max(max_t(S_t) - K, 0)`
- 固定行权价看跌：`max(K - min_t(S_t), 0)`

```python
from derivatives import fixed_strike_lookback_option_price

price = fixed_strike_lookback_option_price(
    S0=100, K=100, T=1, r=0.03, sigma=0.2,
    option_type="call", steps=252, paths=50_000, seed=42,
)
```

离散观察漏掉步间极值，价格通常低于连续观察回望期权。增加 `steps` 会减小这类偏差。

## 5. 多资产篮子期权

到期篮子：

```text
B_T = Σ w_i S_i(T)
payoff = max(B_T - K, 0)
```

```python
from derivatives import basket_option_price

price = basket_option_price(
    spots=[100, 100],
    weights=[0.5, 0.5],
    K=100, T=1, r=0.03,
    volatilities=[0.20, 0.25],
    correlation=[[1.0, 0.4], [0.4, 1.0]],
    dividends=[0.0, 0.02],
    paths=50_000, seed=42,
)
```

相关性越低，算术篮子的波动率一般越低。代码会检查相关矩阵是否对称、对角线是否
为 1、是否半正定，避免生成无效相关随机数。

## 6. 敲入敲出雪球

本实现采用一套常见的简化条款：

1. 每个敲出观察日，若 `S_t/S0 >= knock_out_barrier[t]`，提前兑付  
   `N × (1 + coupon_rate × t)`
2. 未敲出且从未触及敲入线：到期兑付本金 + 完整票息
3. 已敲入且期末低于期初：兑付 `N × S_T/S0`
4. 已敲入但期末回到期初以上：只兑付本金

```python
from derivatives import snowball_note_price

times = [i / 12 for i in range(1, 13)]
result = snowball_note_price(
    S0=100, T=1, r=0.03, sigma=0.2,
    observation_times=times,
    knock_out_barriers=[1.03 - 0.005 * i for i in range(12)],
    knock_in_barrier=0.70,
    coupon_rate=0.12,
    notional=100,
    steps=252, paths=50_000, seed=42,
)
print(result.price, result.autocall_probability)
```

返回 `StructuredNoteResult`：

| 字段 | 含义 |
|------|------|
| `price` | 含本金现值 |
| `standard_error` | MC 价格标准误 |
| `autocall_probability` | 自动赎回概率 |
| `knock_in_probability` | 存续期敲入概率 |
| `loss_probability` | 最终本金受损概率 |
| `expected_life` | 平均兑付时间（年） |

敲入概率不等于亏损概率：敲入后标的仍可能修复到期初以上。

## 7. Phoenix 自动赎回票据

Phoenix 把票息条件与自动赎回条件分开：

- 高于 `coupon_barrier`：支付当期票息
- `memory_coupon=True`：下次满足条件时补发之前未支付的票息
- 高于 `autocall_barrier`：赎回本金
- 未赎回到期且期末低于 `protection_barrier`：本金按标的跌幅承担损失

```python
from derivatives import phoenix_autocall_note_price

times = [i / 12 for i in range(1, 13)]
result = phoenix_autocall_note_price(
    S0=100, T=1, r=0.03, sigma=0.2,
    observation_times=times,
    autocall_barriers=[1.0] * 12,
    coupon_barrier=0.75,
    coupon_rate=0.10,
    protection_barrier=0.65,
    memory_coupon=True,
    steps=252, paths=50_000, seed=42,
)
```

当前保护障碍只在到期判断，不是存续期敲入，因此结果中的
`knock_in_probability` 为 0，`loss_probability` 表示到期跌破保护线且未赎回的概率。

## 8. 参数与条款核对

- [ ] 障碍是相对比例还是绝对价格
- [ ] 敲入按盘中、收盘还是每日最低价
- [ ] 敲出观察日是否使用真实交易日
- [ ] 票息是年化单利、复利还是每期固定金额
- [ ] 敲出当期是否仍支付票息
- [ ] 敲入后上涨是否恢复票息
- [ ] 名义本金、参与率、管理费是否已计入
- [ ] 到期保护是欧式（只看期末）还是美式（存续期观察）

## 9. 运行完整示例

```bash
python3 -m examples.structured_products
python3 -m pytest -q
```

所有 MC 示例固定 `seed` 以便复现。正式估值应提高路径数，并检查价格标准误和参数
压力情景，而不是只报告一个小数点很多的价格。
