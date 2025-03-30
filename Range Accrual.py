import numpy as np
import math

# 可修改参数
S0 =100     #初始资产价格
K = 110     # 目标价格
T = 1.0      # 期限为1年
sigma = 0.2   # 年波动率
r = 0.05     # 年无风险利率
N = 240      # 模拟天数（CN每年交易日）
M = 1000     # 模拟路径数量
lower_bound = 90  # 计息区间的下界
upper_bound = 120 # 计息区间的上界
def generate_path(S0, T, sigma, r, N):
    dt = T/N
    t = np.linspace(0, T, N) #设置等值间隔
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W)*np.sqrt(dt) # 标准布朗运动
    X = (r-0.5 * sigma**2) *t + sigma *W
    S = S0 * np.exp(X)  # 几何布朗运动
    return S

def calculate_payoff(S, lower_bound, upper_bound):
    # 计算在目标范围内的天数
    within_range = np.sum((S > lower_bound) & (S < upper_bound)) # low<s<upper
    return within_range

# 模拟多条路径
payoffs = [] #设置空列表，存放后续的路径
for i in range(M):
    S = generate_path(S0, T, sigma, r, N)
    payoff = calculate_payoff(S, lower_bound, upper_bound)
    payoffs.append(payoff)

# 计算期权的预期值
expected_payoff = np.mean(payoffs)
option_price = np.exp(-r * T) * expected_payoff  # 风险中性定价
print("Estimated Price of the Range Accrual Option:", option_price)