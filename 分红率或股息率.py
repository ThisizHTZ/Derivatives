import numpy as np
import pandas as pd
import openpyxl

# 读取用户上传的数据文件
file_path = '中证500数据.xlsx'
df = pd.read_excel(file_path)

# 重命名列以方便访问
df.columns = ['date', 'index_price', 'future_price', 'near_month_position', 'next_near_month_price', 'next_near_month_position']

# 转换日期列为datetime类型
df['date'] = pd.to_datetime(df['date'])

# 假设每月的近期合约到期日为每月的最后一天
df['future_expiry_date'] = df['date'] + pd.offsets.MonthEnd(0)

# 计算年化基差率
def annualized_basis_rate(row):
    days_to_expiry = (row['future_expiry_date'] - row['date']).days
    if days_to_expiry == 0:
        return np.nan
    return (365 / days_to_expiry) * np.log(row['index_price'] / row['future_price'])

df['annualized_basis_rate'] = df.apply(annualized_basis_rate, axis=1)

# 计算二十日移动平均年化基差率
def moving_avg_basis_rate(t): #公式2
    if t <= 20:
        return np.nan  # 前20天没有足够的数据计算移动平均值
    rates = [(np.log(df.loc[idx, 'index_price'] / df.loc[idx, 'future_price']) * 365) /
             (df.loc[idx, 'future_expiry_date'] - df.loc[idx, 'date']).days
        for idx in range(t-20, t) # 这里设定idx为索引，在20日以前，到当前时间，每20天进行循环
        if (df.loc[idx, 'future_expiry_date'] - df.loc[idx, 'date']).days != 0 ]
    return np.mean(rates) if rates else np.nan  #避免错误，如果是空，就写NaN

df['custom_moving_avg_basis_rate'] = df.index.map(moving_avg_basis_rate)

# 输出结果
df_result = df[['date', '', 'annualized_basis_rate', 'custom_moving_avg_basis_rate']]
print(df_result.head(50))  # 显示前50行，后面很多能显示但是会被省略