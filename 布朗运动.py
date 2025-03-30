import numpy as np
import matplotlib.pyplot as plt
import numba
"""
#  先不用函数的写法写一遍
S0 = 0  # 起点设置
T = 1   # 假定模拟一年的时间
paths = 100 # 假定模拟10条路径
steps = 1000  # 将1年分为100个时间间隔
dt = T / steps # 求出dt
S_path = np.zeros((steps+1,paths))  #创建一个101行（第一行为S0）、10列的矩阵，用来准备储存模拟情况
S_path[0] = S0  #起点设置为S0
rn = np.random.standard_normal(S_path.shape) # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
for step in range(1,steps+1):
    S_path[step] = S_path[step - 1] + rn[step-1]*np.sqrt(dt)
plt.plot(S_path[:,:])
plt.show()"""

def standar_brownian(steps,paths,T,S0): #标准几何布朗运动
    dt = T / steps # 求出dt
    S_path = np.zeros((steps+1,paths))   #创建一个矩阵，用来准备储存模拟情况
    S_path[0] = S0  #起点设置
    rn = np.random.standard_normal(S_path.shape) # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
    for step in range(1,steps+1):
        S_path[step] = S_path[step - 1] + rn[step-1]*np.sqrt(dt)
    plt.plot(S_path[:,:])
    plt.show()
    return S_path
S_path = standar_brownian(steps = 100,paths = 10,T = 1,S0 = 0)
print(S_path)


