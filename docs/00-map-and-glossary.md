# 00 · 地图与术语

## 1. 仓库能力地图

| 你想做的事 | 优先看模块 | 示例入口 |
|------------|------------|----------|
| 欧式香草定价 | `derivatives.pricing.bsm` | `examples.bsm_and_tree` |
| 二叉树对照 BSM | `derivatives.models.binomial_tree` | 同上 |
| 现金或无 / 资产或无 | `derivatives.pricing.binary` | `examples.binary_options` |
| 双障碍 / 二元障碍 | `derivatives.pricing.barrier` | `examples.barrier_and_knock_in` |
| 敲入看涨 | `derivatives.pricing.knock_in` | 同上 |
| 鲨鱼鳍 / 安全气囊 / Range Accrual | `derivatives.pricing.structured` | `examples.structured_products` |
| 蒙特卡洛欧式 | `derivatives.models.monte_carlo` | `examples.monte_carlo` |
| Greeks | `derivatives.analytics.greeks` | `examples.greeks_plot` |
| 中证500基差率 | `derivatives.analytics.basis_rate` | `examples.basis_rate` |
| Wind 取数（可选） | `derivatives.integrations.windlink` | `windlink.py` |

## 2. 符号约定（全库统一）

| 符号 | 含义 | 注意 |
|------|------|------|
| `S` / `S0` | 现价 / 期初价 | 指数点位或现货价格，单位需与 `K` 一致 |
| `K` / `X` | 行权价 | 历史脚本混用，新代码优先 `K`；敲入里 `X` 为行权价、`K` 有时为 rebate |
| `T` | 到期时间（年） | 实务用交易日/365 或实际天数/365，见第 02 章 |
| `r` | 无风险利率（连续） | 国内常近似用国债/资金利率，需明确复利方式 |
| `sigma` | 波动率（年化） | 历史波动率 ≠ 隐含波动率 |
| `q` | 连续股息率 | 股票用分红，指数常用**隐含分红/基差**近似 |
| `b` | 持有成本 | `b=r` 无股利；`b=r-q` 股票；`b=0` 期货期权（Black76 语境） |
| `H` / `upper` / `lower` | 障碍水平 | 连续监控 vs 日终监控价格差很大 |
| `participate_rate` | 参与率 | 结构化产品常见，收益 = 参与率 × 期权收益 |

## 3. 中英术语速查

| 中文 | English | 一句话 |
|------|---------|--------|
| 香草期权 | Vanilla | 标准欧式/美式看涨看跌 |
| 二元期权 | Binary / Digital | 到期要么拿固定金额，要么拿零 |
| 障碍期权 | Barrier | 触及障碍后敲入或敲出 |
| 敲入 / 敲出 | Knock-in / Knock-out | 生效 / 作废条件 |
| 鲨鱼鳍 | Shark Fin | 常见 OTC 结构：未敲出参与上涨，敲出给固定票息 |
| 安全气囊 | Airbag | 下跌保护 + 上行参与的一类结构称呼（条款差异大） |
| 区间计息 | Range Accrual | 标的落在区间内的天数计息 |
| 参与率 | Participation | 对期权收益的缩放 |
| 票息 / 敲出收益 | Coupon / Rebate | 触发障碍后给付的固定收益 |
| 基差 | Basis | 期货 − 现货；指数定价常用来推隐含分红 |
| 希腊字母 | Greeks | 价格对参数的敏感度 |
| 簿记 | Book | 交易台持仓组合 |

## 4. 学习路径建议

1. 会用 BSM，并能复现 put-call parity → 第 01、02 章  
2. 能解释「为什么指数期权用期货基差估分红」→ 第 04 章  
3. 能把鲨鱼鳍拆成「香草 + 障碍 + 二元」→ 第 03 章  
4. 会看 Delta/Vega，知道障碍附近哪里危险 → 第 05 章  
5. 能判断 MC 结果是否靠谱 → 第 06、07 章  

下一章：[01-快速上手](01-quickstart.md)
