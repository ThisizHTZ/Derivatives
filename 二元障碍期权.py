import numpy as np
from scipy.stats import norm

def binary_barrier_option(S, X, H, K, r, b, T, sigma, phi, eta):
        mu = (b - sigma ** 2 / 2) / sigma ** 2
        lambda_ = np.sqrt(mu ** 2 + (2 * r) / sigma ** 2)
        # 计算 x1, x2, y1, y2, z

        x1 = (np.log(S / X) / (sigma * np.sqrt(T))) + (mu + 1) * sigma * np.sqrt(T)
        x2 = (np.log(S / H) / (sigma * np.sqrt(T))) + (mu + 1) * sigma * np.sqrt(T)

        y1 = (np.log(H ** 2 / (S * X)) / (sigma * np.sqrt(T))) + (mu + 1) * sigma * np.sqrt(T)
        y2 = (np.log(H / S) / (sigma * np.sqrt(T))) + (mu + 1) * sigma * np.sqrt(T)

        z =  (np.log(H / S) / (sigma * np.sqrt(T))) + lambda_ * sigma * np.sqrt(T)

        # 计算A和B因素

        A1 = S * np.exp((b - r) * T) * norm.cdf(phi * x1)
        B1 = K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))

        A2 = S * np.exp((b - r) * T) * norm.cdf(phi * x2)
        B2 = K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))

        A3 = S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y1)
        B3 = K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))

        A4 = S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y2)
        B4 = K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))

        A5 = K * ( (H / S) ** (mu + lambda_) * norm.cdf(eta * z) + (H / S) ** (mu - lambda_) * norm.cdf(
            eta * z - 2 * eta * lambda_ * sigma * np.sqrt(T)))

        Downandout_Cashor_Nothing_put = B1-B2 +B3-B4
        return Downandout_Cashor_Nothing_put

# 示例参数
S = 100 #标的资产价格
K = 90 # 行权价
H = 90 #
X = 95 #
r = 0.05
b = 0.02
sigma = 0.2
T = 1
phi= -1
eta= 1
# 计算A和B因素
factors = binary_barrier_option(S, X, H, K, r, b, T, sigma, phi, eta)
print("二元障碍期权在向下敲出的价格为",factors)
