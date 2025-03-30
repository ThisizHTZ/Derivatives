import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats as stats

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #正常显示负号
def geo_brownian(steps,paths,T,S0,b,sigma):
    #用来生成模拟的路径
    dt = T / steps # 时间间隔 dt
    S_path = np.zeros((steps+1,paths))   #创建一个矩阵，用来准备储存模拟情况
    S_path[0] = S0  #起点位置
    #rn = np.random.standard_normal(S_path.shape) #也可以一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
    for step in range(1,steps+1): #创建循环，从1-step+1的位置
        rn = np.random.standard_normal(paths) #创造随机数
        S_path[step] = S_path[step - 1] * np.exp((b-0.5*sigma**2)*dt +sigma*np.sqrt(dt)*rn) #几何布朗运动的解
    return S_path
def MC(steps,paths,T,S0,K,sigma,r,b):
    #根据边界条件求价值
    # b = r为标准的无股利期权，b=r-q为支付股利的期权，b=0为期货期权。
    S_path = geo_brownian(steps,paths,T,S0,b,sigma)  #生成路径
    value = np.exp(-r*T)*np.maximum(S_path[-1]-K,0).mean()  #取模拟的最后价格计算期权价值平均折现
    return value
C = MC(steps = 250,paths = 500000,T = 1,S0 = 100,K=99,sigma = 0.2,r=0.03,b=0.03)
print(C)

