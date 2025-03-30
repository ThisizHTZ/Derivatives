import numpy as np
from scipy.stats import norm
"""
def binary_option_price(S, X1, X2, T, b, r, sigma, option_type='call'):
    d1 = (np.log(S / X1) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * np.exp((b - r) * T) * norm.cdf(d1)) - (X2 * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (X2 * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp((b - r) * T) * norm.cdf(-d1))
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return price

# 参数
S = 50       # 标的资产价格
X1 = 50       # first strike
X2 = 57        # payoff strike
T = 0.5       # 到期时间（年）
b= 0.09     # cost of carry rate
r = 0.09      # 无风险利率
sigma = 0.2   # 波动率 volatility

# 计算看涨二元期权的价格
call_price = binary_option_price(S, X1, X2, T, b, r, sigma, option_type='call')
put_price = binary_option_price(S, X1, X2, T, b, r, sigma, option_type='put')
print("Binary gap option 缺口期权价格:", call_price)

#现金或无期权。在到期日，若标的资产价格高于或低于约定的执行价格，获得固定的现金收益，否则收益为0。
def Cash_or_Nothing_Call(S, X,K, T, r,b, sigma, option_type='call'):
    d1 = (np.log(S / X) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    c = K*np.exp(-r * T) * norm.cdf(d1)
    return c

def Cash_Or_Nothing_Put(S, X,K, T, r,b, sigma, option_type='put'):
    d1 = (np.log(S / X) + (b - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    p = K * np.exp(-r * T) * norm.cdf(-d1)
    return p

S=100
X=80
K=10
T=0.75
r=0.06
b=0
sigma=0.35

call_price = Cash_or_Nothing_Call(S, X,K, T, r,b, sigma, option_type='call')
put_price = Cash_Or_Nothing_Put(S, X,K, T, r,b, sigma, option_type='put')

print(f" Cash_or_Nothing_Call 现金或无现金看涨期权价格: {call_price:.5f}")
print(f"Cash_Or_Nothing_Put 现金或无现金看跌期权价格: {put_price:.5f}")

def Asset_Or_Nothing_Call(S, X,K,T,r, b, sigma, option_type= 'call'):
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    c = K * np.exp(-r * T) * norm.cdf(d1)
    return c
def Asset_or_Nothing_Put(S, X, K, T, r, b, sigma, option_type='put'):
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    c = K * np.exp(-r * T) * norm.cdf(-d1)
    return c

S=70
X=65
K=10
T=0.5
r=0.07
b=0.02 #b=r-q, 0.07-0.05=0.02,注意这里不会使用基差率！b只是他们的差
sigma=0.27

call_price = Asset_Or_Nothing_Call(S, X, K, T, r, b, sigma, option_type='call')
put_price = Asset_or_Nothing_Put(S, X,K, T, r,b, sigma, option_type='put')

print(f" Asset_Or_Nothing_Call 资产或无资产看涨期权价格: {call_price:.5f}")
print(f"Asset_Or_Nothing_Put 资产或无资产看跌期权价格: {put_price:.5f}")
"""

#这是supershare option，用处不大，作为卖方需要风险对冲。
def supershare_option_price(S, X1, X2, T, r, b, sigma):

    d1 = (np.log(S / X1) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / X2) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price1 = np.exp(-r * T) * norm.cdf(d1)
    price2 = np.exp(-r * T) * norm.cdf(d2)
    w = ((S * np.exp(b - r) * T) / X1) * (norm.cdf(d1) - norm.cdf(d2))
    return price1, price2, w

# 示例参数
S = 100    # 标的资产价格
X1 = 90    # X1= Xl, 也就是较低行权价
X2 = 110   # X2= XH, 也就是较高行权价
T = 0.25   # 到期时间（年）
r = 0.1    # 无风险利率
b = 0      # 持有成本率
sigma = 0.2 # 波动率

# 计算 Supershare Options 的价格
supershare_price = supershare_option_price(S, X1, X2, T, r, b, sigma)
print("Supershare option 价格:", supershare_price)

