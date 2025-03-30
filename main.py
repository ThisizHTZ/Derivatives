import numpy as np
import math
import matplotlib
from scipy.stats import norm

def shark_fin_option(S, K, T, r, sigma, up, steps, simulations, option_type='call'):
    """
    单向鲨鱼鳍期权的价格。

    参数:
    S : float
        标的资产的当前价格
    K : float
        行权价格
    T : float
        到期时间（以年为单位）
    r : float
        无风险利率
    sigma : float
        年化波动率
    up : float
        上障碍
    steps : int
        时间步数
    simulations : int
        模拟次数
    option_type : str
        期权类型（'call'表示看涨，'put'表示看跌）
    返回:
    float
        期权的估计价格
    """
    dt = T / steps  # 每步的时间间隔
    expirations = np.zeros(simulations)
    for i in range(simulations):
        prices = np.exp(np.log(S) +np.cumsum((r - 0.5 *sigma **2)*dt + sigma * np.sqrt(dt) *np.random.normal(size=steps)))
        # 检查是否触及障碍
        if np.max(prices) >= up:
            if option_type == 'call':
                expirations[i] = max(prices[-1] - K, 0)
            else:
                expirations[i] = max(K - prices[-1], 0)
    # 计算折现后的期权价值
    return np.exp(-r * T) * np.mean(expirations)

# 示例参数
S = 100  # 标的资产价格
K = 100  # 行权价格
T = 1  # 到期时间（年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
B_up = 120  # up barrier
steps = 100  # 时间步长
simulations = 10000  # 模拟次数

# 计算看涨期权的价格
call_price = shark_fin_option(S, K, T, r, sigma, B_up, steps, simulations, 'call')
put_price = shark_fin_option(S, K, T, r, sigma, B_up, steps, simulations, 'put')
print("单向鲨鱼鳍看涨期权价格:", call_price)
print("单向鲨鱼鳍看跌期权价格:", put_price)


from scipy.stats import norm

def double_barrier_call_option(S, K, T, r, sigma, U, L, max_iterations=100):
    """
    使用数值方法计算双障碍看涨期权的价格。

    需要的参数:
    S : float - 标的资产的当前价格
    K : float - 行权价格
    T : float - 到期时间（年）
    r=b : float - 无风险利率
    sigma : float - 波动率
    U : float - 上障碍
    L : float - 下障碍
    max_iterations : int - 用于无限级数的最大迭代次数

    返回:
    float - 期权的估计价格
    """
    dt = sigma * np.sqrt(T)
    price = 0

    for n in range(-max_iterations, max_iterations+1):
        factor1 = (U/ S)** (2*n)
        factor2 = (L/S)** (2*n)
        d1 = (np.log(S * factor1 / K) + (r + 0.5 * sigma**2) * T) / dt
        d2 = d1 - dt
        d3 = (np.log(S * factor2 / K) + (r + 0.5 * sigma**2) * T) / dt
        d4 = d3 - dt

        term1 = factor1* (norm.cdf(d1)- norm.cdf(d2))
        term2 = factor2 *(norm.cdf(-d3)- norm.cdf(-d4))
        price += term1 - term2

    return np.exp(-r * T) * price

# 示例参数
S = 100       # 标的资产价格
K = 90       # 行权价格
T = 1         # 到期时间（年）
r = 0.05      # 无风险利率
sigma = 0.2   # 波动率
U = 120       # 上障碍
L = 80        # 下障碍

# 计算看涨期权的价格
call_price = double_barrier_call_option(S, K, T, r, sigma, U, L)
print("双障碍看涨期权价格:", call_price) #Double Barrier Options

"""使用数值方法计算双障碍看跌期权的价格。

    参数:
    S : float - 标的资产的当前价格
    K : float - 行权价格
    T : float - 到期时间（年）
    r : float - 无风险利率
    sigma : float - 波动率
    U : float - 上障碍
    L : float - 下障碍
    max_iterations : int - 用于无限级数的最大迭代次数

    返回:
    float - 期权的估计价格
"""

def double_barrier_put_option(S, K, T, r, sigma, U, L, max_iterations=100):
    dt = sigma * np.sqrt(T)
    price = 0

    for n in range(-max_iterations, max_iterations + 1):
        factor1 = (U / S) ** (2 * n)
        factor2 = (L / S) ** (2 * n)

        d1 = (np.log(S * factor1 / K) + (r + 0.5 * sigma ** 2) * T) / dt
        d2 = d1 - dt
        d3 = (np.log(S * factor2 / K) + (r + 0.5 * sigma ** 2) * T) / dt
        d4 = d3 - dt

        term1 = factor1 * (norm.cdf(-d1) - norm.cdf(-d2))
        term2 = factor2 * (norm.cdf(d3) - norm.cdf(d4))

        price += term1 - term2

    final_price = np.exp(-r * T) * price
    return final_price if final_price >= 0 else 0  # 确保返回非负值


# 示例参数
S = 100  # 标的资产价格
K = 90  # 行权价格
T = 1  # 到期时间（年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
U = 120  # 上障碍
L = 80  # 下障碍

# 计算看跌二元期权的价格
put_price = double_barrier_put_option(S, K, T, r, sigma, U, L)
print("双障碍看跌期权价格:", put_price)

import numpy as np
from scipy.stats import norm

def binary_option_price(S, K, T, r, sigma, option_type='call'):
    d = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        price = np.exp(-r * T) * norm.cdf(d)
    elif option_type == 'put':
        price = np.exp(-r * T) * norm.cdf(-d)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return price

# 示例参数
S = 100       # 标的资产价格
K = 90       # 行权价格
T = 1         # 到期时间（年）
r = 0.05      # 无风险利率
sigma = 0.2   # 波动率

# 计算看涨二元期权的价格
call_price = binary_option_price(S, K, T, r, sigma, 'call')
print("二元期权看涨二元期权价格:", call_price)

import numpy as np
from scipy.stats import norm

def double_barrier_call_option(S, K, T, r, sigma, U, L, max_iterations=100):
    """
    使用数值方法计算双障碍看涨期权的价格。

    需要的参数:
    S : float - 标的资产的当前价格
    K : float - 行权价格
    T : float - 到期时间（年）
    r=b : float - 无风险利率
    sigma : float - 波动率
    U : float - 上障碍
    L : float - 下障碍
    max_iterations : int - 用于无限级数的最大迭代次数

    返回:
    float - 期权的估计价格
    """
    dt = sigma * np.sqrt(T)
    price = 0

    for n in range(-max_iterations, max_iterations+1):
        factor1 = (U/ S)** (2*n)
        factor2 = (L/S)** (2*n)
        d1 = (np.log(S * factor1 / K) + (r + 0.5 * sigma**2) * T) / dt
        d2 = d1 - dt
        d3 = (np.log(S * factor2 / K) + (r + 0.5 * sigma**2) * T) / dt
        d4 = d3 - dt

        term1 = factor1* (norm.cdf(d1)- norm.cdf(d2))
        term2 = factor2* (norm.cdf(-d3)- norm.cdf(-d4))
        price += term1 - term2

    return np.exp(-r * T) * price

# 示例参数
S = 100       # 标的资产价格
K = 90      # 行权价格
T = 1        # 到期时间（年）
r = 0.03      # 无风险利率
sigma = 0.3    # 波动率
U = 130       # 上障碍
L = 80        # 下障碍

# 计算看涨期权的价格
call_price = double_barrier_call_option(S, K, T, r, sigma, U, L)
print("双障碍看涨期权价格:", call_price) #Double Barrier Options


def barrier_option_price(S, K, T, r, sigma, U, L, delta):
    b = r - delta  # 考虑股息率
    E = L * np.exp(delta * T)  # 根据公式定义E
    mu = 2 * (b - delta) / sigma ** 2
    delta1 = mu + sigma ** 2 / (2 * b)
    delta2 = 2 * b / sigma ** 2
    sum1 = 0
    sum2 = 0

    max_iterations = 50  # 可以调整迭代次数以优化精度与性能
    for n in range(-max_iterations, max_iterations):
        U_n = U * (L / U) ** (2 * n)
        L_n = L * (U / L) ** (2 * n)

        d1 = (np.log(S * U_n / (E * L_n)) + (b + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        d3 = (np.log(L_n ** 2 / (S * U_n * K)) + (b + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d4 = d3 - sigma * np.sqrt(T)

        sum1 += (U_n / L_n) ** n * (L / S) ** (2 * n) * (norm.cdf(d1) - norm.cdf(d2))
        sum2 += (L_n / U_n) ** (n + 1) * (U_n / S) ** (2 * n) * (norm.cdf(-d3) - norm.cdf(-d4))

    out_value = S * np.exp((b - r) * T) * sum1 - K * np.exp(-r * T) * sum2
    return out_value


# 示例参数
S = 100  # 标的资产价格
K = 100  # 行权价格
T = 1  # 到期时间（年）
r = 0.05  # 无风险利率
sigma = 0.20  # 波动率
U = 120  # 上障碍
L = 80  # 下障碍
delta = 0.03  # 股息率

# 计算双向障碍看涨期权的价格
option_price = barrier_option_price(S, K, T, r, sigma, U, L, delta)
print("Double Barrier Call Option Price: {:.2f}".format(option_price))
