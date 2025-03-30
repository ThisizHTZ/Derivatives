import numpy as np
from scipy.stats import norm
import math
def Up_and_out_call(S, X, T, r, q, sigma, H, K, participate_rate, gr):
    b = r - q # 无风险利率-分红率
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    lamb = math.sqrt(mu ** 2 + (2 * r / (sigma ** 2)))

    x1 = (math.log(S / X) / (sigma * math.sqrt(T))) + ((1 + mu) * sigma * math.sqrt(T))
    x2 = (math.log(S / H) / (sigma * math.sqrt(T))) + ((1 + mu) * sigma * math.sqrt(T))
    y1 = ((math.log(H ** 2 / (S * X))) / (sigma * math.sqrt(T))) + ((1 + mu) * (sigma * math.sqrt(T)))
    y2 = (math.log(H / S) / (sigma * math.sqrt(T))) + ((1 + mu) * (sigma * math.sqrt(T)))
    z = (math.log(H / S) / (sigma * (math.sqrt(T))) + lamb * sigma * math.sqrt(T))

    A = S * math.exp((b - r) * T) * norm.cdf(x1) - X * math.exp(-r * T) * norm.cdf(x1 - sigma * math.sqrt(T))
    B = S * math.exp((b - r) * T) * norm.cdf(x2) - X * math.exp(-r * T) * norm.cdf(x2 - sigma * math.sqrt(T))
    C = S * math.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y1) - X * math.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(-y1 + sigma * math.sqrt(T))
    D = S * math.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y2) - X * math.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(-y2 + sigma * math.sqrt(T))
    E = K * math.exp(-r * T) * (norm.cdf(-x2 + sigma * math.sqrt(T)) - (H / S) ** (2 * mu) * norm.cdf(-y2 + sigma * math.sqrt(T)))
    F = K * (((H / S) ** (mu + lamb) * norm.cdf(-z)) + (H / S) ** (mu - lamb) * norm.cdf(-z + 2 * lamb * sigma * math.sqrt(T)))

    C_uo_XltH = max(A - B + C - D + F, gr) * participate_rate
    return C_uo_XltH

# 设置参数
S = 100  # 标的价格
X = 92  # 行权价格
T = 91/365  # 到期时间（年）
r = 0.02  # 无风险利率
q = 0  # 分红率
sigma = 0.07  # 波动率
H = 109  # 障碍水平
K = 2.45  # 敲出收益
participate_rate = 0.2647 #参与率
gr = 0.5 #保底收益

# 计算上障碍看涨期权的收益
result = Up_and_out_call(S, X, T, r, q, sigma, H, K, participate_rate, gr)
print("如果行权价格未超过障碍水平，期权价格为", result)