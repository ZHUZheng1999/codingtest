# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
signal_df = pd.DataFrame()
predict_df = pd.read_csv('./predict_data_all.csv')

signal_df['date'] = predict_df['date']
list_data = [0 for i in range(len(predict_df['date']))]
signal_df.set_index('date',drop=True,inplace=True)
predict_df.set_index('date',drop=True,inplace=True)
folder_name = os.listdir('./predict')
index = 0
for name in folder_name:
    stock = name[:-4]
    signal_df.insert(loc=index,column = stock, value = list_data)
    index+=1
    
for i in signal_df.index:
    for stock in predict_df.columns:
        if predict_df.loc[i,stock] > np.percentile(np.array(predict_df.loc[i,:]),90):
            signal_df.loc[i,stock] = 1
        elif predict_df.loc[i,stock] < np.percentile(np.array(predict_df.loc[i,:]),10):
            signal_df.loc[i,stock] = -1
        else:
            signal_df.loc[i,stock] = 0

return_ratio = pd.DataFrame()
return_ratio['date'] = signal_df.index
return_ratio.set_index('date',drop=True,inplace=True)
return_ratio['return'] = 0
true_df = pd.read_csv('./true_df.csv')
true_df.set_index('date',drop=True,inplace=True)
for i in signal_df.index:
    a = []
    b = []
    for stock in signal_df.columns:
        if signal_df.loc[i,stock] == 1:
            a.append(true_df.loc[i,stock]/100-0.0016)
        elif signal_df.loc[i,stock] == -1:
            b.append((-1)*true_df.loc[i,stock]/100-0.0016)
        return_ratio.loc[i,'return'] = np.mean(a+b)*0.95
    if len(a+b) == 0:
        return_ratio.loc[i,'return'] = 0
return_ratio['asset'] = 0
return_ratio.iloc[0,1] = return_ratio.iloc[0,0]+1
for i in range(1,len(return_ratio['return'])):
    return_ratio.iloc[i,1] =  (return_ratio.iloc[i,0]+1)*return_ratio.iloc[i-1,1]
print(return_ratio)

plt.figure()
plt.plot(return_ratio['asset'],label="Asset")
plt.show()
plt.savefig('./asset.png')

def Maxdrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list)-return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])
    drawdown_max = return_list[j]-return_list[i]
    drawdown_rate = 1- return_list[i]/return_list[j]
    drawdown_day = i-j
    return drawdown_max, drawdown_rate, drawdown_day

print("totol return is:", return_ratio.iloc[-1,1]-1)
A = Maxdrawdown(return_ratio.iloc[:,1])
print("Max drawdown rate is :", A)

sharpe_ratio = np.mean(return_ratio['return'])/np.std(return_ratio['return'])*math.sqrt(252)
print("sharpe ratio is :", sharpe_ratio)

vol = return_ratio['return'].std()*math.sqrt(252)
print("volatility is :",vol)
    

        