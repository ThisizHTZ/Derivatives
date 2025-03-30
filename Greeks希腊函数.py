import numpy as np
from numpy import exp  #直接导入exp，省的每次都要写np.exp
from scipy.stats import norm #直接导入norm，省的每次都要写stats.norm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #正常显示负号

def greeks(CP,S,X,sigma,T,r,b): #计算greeks的函数
    """
    Parameters
    ----------
    CP：看涨或看跌"C"or"P"
    S : 标的价格.
    X : 行权价格.
    sigma :波动率.
    T : 年化到期时间.
    r : 无风险收益率.
    b : 持有成本，当b = r 时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
    Returns
    -------
    返回欧式期权的估值和希腊字母
    """
    d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma* np.sqrt(T)) #求d1
    d2 = d1 - sigma * np.sqrt(T) #求d2

    if CP == "C":
        option_value = S * exp((b - r)*T) * norm.cdf(d1) - X * exp(-r*T) * norm.cdf(d2) #计算期权价值
        delta = exp((b-r)*T)* norm.cdf(d1)
        gamma = exp((b-r)*T)* norm.pdf(d1)/ (S * sigma * T**0.5) #注意是pdf，概率密度函数
        vega = S * exp((b-r)*T) * norm.pdf(d1) * T**0.5 # 计算vega
        theta =  -exp((b-r)*T) * S * norm.pdf(d1) * sigma / (2*T**0.5) -r*X*exp(-r*T)*norm.cdf(d2) - (b-r) * S * exp((b-r)*T)*norm.cdf(d1)
        if b !=0: #rho比较特别，b是否为0会影响求导结果的形式
            rho = X * T * exp(-r*T) * norm.cdf(d2)
        else:
            rho = -T * exp(-r*T) * (S*norm.cdf(d1)-X*norm.cdf(d2))
        return rho
    else:
        option_value =  X * exp(-r*T) * norm.cdf(-d2) - S * exp((b - r)*T) * norm.cdf(-d1)
        delta = -exp((b-r)*T)* norm.cdf(-d1)
        gamma = exp((b-r)*T)* norm.pdf(d1)/ (S * sigma * T**0.5) #跟看涨其实一样，不过还是先写在这里
        vega = S * exp((b-r)*T) * norm.pdf(d1) * T**0.5 # #跟看涨其实一样，不过还是先写在这里
        theta =  -exp((b-r)*T) * S * norm.pdf(d1) * sigma / (2*T**0.5) + r*X*exp(-r*T)*norm.cdf(-d2) + (b-r) * S * exp((b-r)*T)*norm.cdf(-d1)
        if b !=0: #rho比较特别，b是否为0会影响求导结果的形式
            rho = -X * T * exp(-r*T) * norm.cdf(-d2)
        else:
            rho = -T * exp(-r*T) * (X*norm.cdf(-d2) - S*norm.cdf(-d1))
    #  写成函数时要有个返回，这里直接把整个写成字典一次性输出。
    greeks = {"option_value":option_value,"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}
    return greeks

greeks(CP,S,X,sigma,T,r,b)
{'option_value': 5.046872445344206,
    'delta': 0.49241608901083517,
    'gamma': 0.028070194936741403,
    'vega': 27.511598057500255,
    'theta': -5.350913438139725,
    'rho': -2.5234362226721037}


#下面使用greeks函数简单画图分析一下，因为希腊字母的分析还是比较复杂的，随到期时间、实值虚值等均有所不同，这里只做简单列示：#在不同价格下，希腊字母的变化，当然也可以很方便的画出其他参数变化下的希腊字母变化，这里就不画那么多了
S = np.linspace(0.1,200,100) #生产0.01到200的100个价格序列
result = greeks(CP,S,X,sigma,T,r,b)
fig,ax = plt.subplots(nrows=3,ncols=2,figsize = (8,12)) #使用多子图的方式输入结果，所以写的复杂一点
greek_list = [['option_value','delta'],['gamma','vega'],['theta','rho']] #和子图的二维数组对应一下
for m in range(3):
    for n in range(2):
        plot_item = greek_list[m][n]
        ax[m,n].plot(S,result[plot_item])
        ax[m,n].legend([plot_item])

