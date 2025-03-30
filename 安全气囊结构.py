import numpy as np
from scipy.stats import norm

def Vanilla_at_the_money_Call(S, K, H, T, r, sigma,participate_rate): #香草看涨期权
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def down_and_in_put_price(S, K, H, T, r, sigma,participate_rate): #向下敲入看跌期权
    """
    S: 当前股票价格
    K: 行权价格
    H: 敲入价格水平
    T: 到期时间（年）
    r: 无风险利率
    sigma: 年化波动率
    """
    lambda_ = (r + 0.5 * sigma ** 2) / sigma ** 2
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    y = np.log(H / S) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)

    # 使用公式计算向下敲入看跌期权的价格
    price = K * np.exp(-r * T) * norm.cdf(y) - S * norm.cdf(y - sigma * np.sqrt(T)) - \
            K * np.exp(-r * T) * (H / S)**(2 * lambda_) * norm.cdf(y1) + \
            S * (H / S)**(2 * lambda_) * norm.cdf(y1 - sigma * np.sqrt(T))

    return price

def airbag_option_price(S, K, H, T, r, sigma,participate_rate):
    vanilla_put = Vanilla_at_the_money_Call(S, K, H, T, r, sigma,participate_rate)
    barrier_put = down_and_in_put_price(S, K, H, T, r, sigma,participate_rate)
    airbag_price = (vanilla_put + barrier_put) * participate_rate
    return airbag_price

# 示例参数
S = 100    # 标的资产价格
K = 100     # 行权价
H = 70     # 障碍价
T = 1      # 到期时间（年）
r = 0.05   # 无风险利率
sigma = 0.2 # 波动率
participate_rate= 0.7
maxprice = 108

# 计算安全气囊期权的价格
airbag_price = airbag_option_price(S, K, H, T, r, sigma,participate_rate)
print(f"Airbag option 价格: {airbag_price:.5f}")
