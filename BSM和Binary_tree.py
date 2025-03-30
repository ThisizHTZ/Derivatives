import numpy as np
import math
import matplotlib
from scipy.stats import norm

# 定义BSM模型函数
def Black_scholes_merton(S, K, T, r, sigma, option_type='call'):
    #计算欧式期权的价格。

    #参数:
    #S : float
        #标的资产当前价格
    #K : float
        #行权价格 price
    #T : float
        #到期时间（我们以year为单位）
    #r : float
        #无风险利率（年化）
    #sigma=vol: float
        #historical volatility #P = f(S,K,T,Vol,R)
    #option_type : str
        #期权类型，'call'买 或 'put'卖

    #norm.cdf: 累计分布函数在正态分布上出现的概率
    #np.exp(): 表示e的（）次方
    #np.sqrt()：开方
    #返回:float 也就是期权的理论价格

    # 计算d1和d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # 根据期权类型计算价格
    if option_type == 'call':
        price = (S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return price

# mock参数
S = 100 # 标的资产当前价格
K = 120  # 行权价格
T = 0.5  # 到期时间（一年）
r = 0.05 # 无风险利率（5%）
sigma = 0.2  # 波动率（20%）

# 计算期权价格
call_price = Black_scholes_merton(S, K, T, r, sigma, 'call')
put_price = Black_scholes_merton(S, K, T, r, sigma, 'put')

print("BSM Call Option Price: {:.3f}".format(call_price))
print("BSM Put Option Price: {:.3f}".format(put_price))

#这是一个欧式看涨期权的二叉树模型

def tree_europ(S,X,r,sigma,t,steps):
    u=np.exp(sigma*np.sqrt(t/steps));d=1/u #注意时间间隔为△t=t/steps
    P=(np.exp(r*t/steps)-d)/(u-d)
    prices=np.zeros(steps+1) #生成最后一列的股票价格空数组
    c_values=np.zeros(steps+1) #生成最后一列的期权价值空数组
    prices[0]=S *d ** steps #最后一行最后一列的股票价格
    c_values[0]=np.maximum(prices[0]-X,0) #最后一行最后一列的期权价值

    for i in range(1,steps+1):
        prices[i]=prices[i-1]*(u**2) #计算最后一列的股票价格
        c_values[i]=np.maximum(prices[i]-X,0) #计算最后一列的期权价值
    for j in range(steps,0,-1): #逐个节点往前计算
        for i in range(0,j):
            c_values[i]=(P*c_values[i+1]+(1-P)*c_values[i])/np.exp(r*t/steps)
    return c_values[0]

print(tree_europ(100, 120, 0.05, 0.2, 0.5, 100))  # 应该输出期权的理论价值

#二叉树方法
import numpy as np

def EurPut(S, K, r, q, sigma, t, steps):
    # 计算上升和下降因子
    u = np.exp(sigma * np.sqrt(t / steps))
    d = 1 / u
    # 计算风险中性概率
    p = (np.exp((r - q) * t / steps) - d) / (u - d)
    price_steps_term = []
    option_values = []

    # 初始化期权的末端价值
    for i in range(steps + 1):
        price_steps_term.append(S * u ** (steps - 2 * i))
        option_values.append(max(K - price_steps_term[i], 0))

    # 从末期向前递推期权价值
    for j in range(steps, 0, -1):
        for l in range(j):
            option_values[l] = (p * option_values[l] + (1 - p) * option_values[l + 1]) * np.exp(-r * t / steps)

    # 返回期权的当前估价
    return option_values[0]

# 设置参数
S = 100  # 标的资产当前价格
K = 120  # 行权价格
T = 0.5  # 到期时间（半年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
q = 2     # 股息率
steps = 1000  # 时间步长

# 计算期权价格并打印
option_price = EurPut(S, K, r, q, sigma, T, steps)
print("二叉树期权定价: {:.2f}".format(option_price))



