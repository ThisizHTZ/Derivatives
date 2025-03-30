import numpy as np
from scipy.stats import norm
import math

def european_call_option(S, X, T, r, sigma): #看涨欧式期权公式
    d1 = (math.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)

#向下敲入看涨
def down_and_in_call(S, X, T, r, sigma, H, K):
    mu = (r - 0.5 * sigma ** 2) / sigma ** 2
    lambda_ = math.sqrt(mu ** 2 + 2 * r / sigma ** 2)

    x1 = (math.log(S / X) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    x2 = (math.log(S / H) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y1 = (math.log(H ** 2 / (S * X)) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y2 = (math.log(H / S) + (1 + mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    z = (math.log(H / S) + lambda_ * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))

    A = S * math.exp((r - sigma ** 2 / 2) * T) * norm.cdf(x1) - X * math.exp(-r * T) * norm.cdf(x1 - sigma * math.sqrt(T))
    B = S * math.exp((r - sigma ** 2 / 2) * T) * norm.cdf(x2) - X * math.exp(-r * T) * norm.cdf(x2 - sigma * math.sqrt(T))
    C = S * math.exp((r - sigma ** 2 / 2) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(y1) - X * math.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(y1 - sigma * math.sqrt(T))
    D = S * math.exp((r - sigma ** 2 / 2) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(y2) - X * math.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(y2 - sigma * math.sqrt(T))
    E = K * math.exp(-r * T) * (norm.cdf(-z) - (H / S) ** (2 * mu) * norm.cdf(y2 - sigma * math.sqrt(T)))
    F = K * ((H / S) ** (mu + lambda_) * norm.cdf(z) + (H / S) ** (mu - lambda_) * norm.cdf(z - 2 * lambda_ * sigma * math.sqrt(T)))

    c_di_XmtH = C + E ##More than
    c_di_XltH = A - B + D + E #Less than

    return c_di_XmtH, c_di_XltH

#向上敲入看涨期权 书里的barrier公式
def Up_and_in_call(S, X, T, r, sigma, H, K):
    mu = (r - 0.5 * sigma ** 2) / sigma ** 2
    lambda_ = math.sqrt(-mu ** 2 + 2 * r / sigma ** 2)

    x1 = (math.log(S / X) + (1 -mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    x2 = (math.log(S / H) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y1 = (math.log(H ** 2 / (S * X)) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    y2 = (math.log(H / S) + (1 - mu) * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))
    z = (math.log(H / S) + lambda_ * sigma * math.sqrt(T)) / (sigma * math.sqrt(T))

    A = S * math.exp((r - sigma ** 2 / 2) * T) * norm.cdf(x1) - X * math.exp(-r * T) * norm.cdf(x1 - sigma * math.sqrt(T))
    B = S * math.exp((r - sigma ** 2 / 2) * T) * norm.cdf(x2) - X * math.exp(-r * T) * norm.cdf(x2 - sigma * math.sqrt(T))
    C = S * math.exp((r - sigma ** 2 / 2) * T) * (H / S) ** (2 * (-mu + 1)) * norm.cdf(y1) - X * math.exp(-r * T) * (H / S) ** (2 * -mu) * norm.cdf(y1 - sigma * math.sqrt(T))
    D = S * math.exp((r - sigma ** 2 / 2) * T) * (H / S) ** (2 * (-mu + 1)) * norm.cdf(y2) - X * math.exp(-r * T) * (H / S) ** (2 * -mu) * norm.cdf(y2 - sigma * math.sqrt(T))
    E = K * math.exp(-r * T) * (norm.cdf(-z) - (H / S) ** (2 * -mu) * norm.cdf(y2 - sigma * math.sqrt(T)))
    F = K * ((H / S) ** (-mu + lambda_) * norm.cdf(z) + (H / S) ** (-mu - lambda_) * norm.cdf(z - 2 * lambda_ * sigma * math.sqrt(T)))

    c_ui_XmtH = A+E
    c_ui_XltH = B-C+D+E
    return c_ui_XmtH,c_ui_XltH

# 设置参数
S = 105  # 股票价格
X = 95  # 行权价格
T = 1  # 到期时间（年）
r = 0.04  # 无风险利率
sigma = 0.5  # 波动率
H = 90  # 障碍水平
K= 0.02

# 计算 Down-and-in 看涨期权的收益
result1 = down_and_in_call(S, X, T, r, sigma, H, K)
print("Down-and-in 向下敲入看涨生效期权：", result1)

# 计算 Up-and-in 看涨期权的收益
result2 = Up_and_in_call(S, X, T, r, sigma, H, K)
print("Up-and-in 向上敲入看涨生效期权：", result2)
