
import pandas as pd
import numpy as np
import math
import numba  #numba加速运算的大杀器，后面用到
def simulate_tree_am(CP,m,S0,T,sigma,K,r,b):  # 美式期权二叉树模型
    """
    CP:看涨或看跌
    m：模拟的期数
    S0：期初价格
    T：期限
    sigma：波动率
    K：行权价格
    r:无风险利率
    b:持有成本,当b = r 时，为标准的无股利模型，b=0时，为black76，b为r-q时，为支付股利模型，b为r-rf时为外汇期权
    """
    dt = T/m
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/u
    S = np.zeros((m+1,m+1))
    S[0,0] = S0
    p = (math.exp(b*dt) - d)/(u-d)
    for i in range(1,m+1):  #模拟每个节点的价格
        for a in range(i):
            S[a,i] = S[a,i-1] * u
            S[a+1,i] = S[a,i-1] * d
    Sv = np.zeros_like(S) #创建期权价值的矩阵，用到从最后一期倒推期权价值
    if CP == "C":
        S_intrinsic = np.maximum(S-K,0)
    else:
        S_intrinsic = np.maximum(K-S,0)
    Sv[:,-1] = S_intrinsic[:,-1]
    for i in range(m-1,-1,-1): #反向倒推每个节点的价值
        for a in range(i+1):
            Sv[a,i] = max((Sv[a,i+1] * p + Sv[a+1,i+1] * (1-p))/np.exp(r*dt),S_intrinsic[a,i])
    return Sv[0,0]
simulate_tree_nb = numba.jit(simulate_tree_am)
val = simulate_tree_am(CP = "C",m=1000,S0 = 100,K = 95,sigma = 0.25,T = 1,r = 0.03,b = 0.03)

print("二叉树美式期权定价",val)