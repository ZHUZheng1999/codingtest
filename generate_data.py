# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
df = pd.read_csv('D:/coding_test/data/data.csv')
stock_list = set(df['ticker'])
A = {}
for stock in stock_list:
    A[stock] = df[df['ticker'] == stock]
    A[stock].to_csv('D:/coding_test/raw_stock/'+stock+'.csv')

def calculateEMA(period,closeArray,emaArray=[]):
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan],(nanCounter+period-1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter+period-1])
        emaArray.append(firstema)
        for i in range(nanCounter+period,length):
            ema = (2*closeArray[i]+(period-1)*emaArray[-1])/(period+1)
            emaArray.append(ema)
    return np.array(emaArray)

def calculateMACD(closeArray,shortperiod = 10, longperiod = 30, signalperiod = 15):
    ema10 = calculateEMA(shortperiod, closeArray,[])
    ema30 = calculateEMA(longperiod, closeArray,[])
    diff = ema10 - ema30
    dea = calculateEMA(signalperiod, diff,[])
    macd = diff - dea
    fast_values = diff
    slow_values = dea
    diff_values = macd
    return fast_values, slow_values,diff_values

def RSI(array_list,periods = 20):
    length = len(array_list)
    rsies = [np.nan]*length
    if length <= periods:
        return rsies
    up_avg = 0
    down_avg = 0
    first_t = array_list[:periods+1]
    for i in range(1,len(first_t)):
        if first_t[i] >= first_t[i-1]:
            up_avg+=first_t[i]-first_t[i-1]
        else:
            down_avg+=first_t[i-1]-first_t[i]
    up_avg = up_avg/periods
    down_avg = down_avg/periods
    rs = up_avg/down_avg
    rsies[periods] = 100-100/(1+rs)
    
    for j in range(periods+1,length):
        up= 0
        down = 0
        if array_list[j]>=array_list[j-1]:
            up = array_list[j] - array_list[j-1]
            down = 0
        else:
            down = array_list[j-1] - array_list[j]
            up=0
        up_avg = (up_avg*(periods-1)+up)/periods
        down_avg = (down_avg*(periods-1)+down)/periods
        rs = up_avg/down_avg
        rsies[j] = 100-100/(1+rs)
    return rsies

def getPSY(pricedata, period = 20):
    diff = pricedata[1:]-pricedata[:-1]
    diff = np.append(0,diff)
    diff_dir = np.where(diff>0, 1, 0)
    psy = np.zeros((len(pricedata),))
    psy[:period] *= np.nan
    for i in range(period,len(pricedata)):
        psy[i] = (diff_dir[i-period+1].sum())/period
    return psy*100


Dict = {}
folder_name = os.listdir('./raw_stock')
for name in folder_name:
    stock = name[:-4]
    df_main = pd.read_csv('./raw_stock/'+name,usecols=[2,3,4])
    df_main.set_index('date',drop = True, inplace = True)
    alpha1 = {}
    for i in range(len(df_main)):
        if i <= 3:
            alpha1[i] = 0
        else:
            alpha1[i] = -np.std(df_main.iloc[i-4:i+1,1])
    df_main['Alpha1'] = alpha1.values()
    alpha2 = {}
    for i in range(len(df_main)):
        if i <= 4:
            alpha2[i] = 0
        else:
            alpha2[i] = df_main.iloc[i,0]/df_main.iloc[i-5,0]
    df_main['Alpha2'] = alpha2.values()
    alpha3 = {}
    for i in range(len(df_main)):
        if i <= 5:
            alpha3[i] = 0
        else:
            alpha3[i] = (df_main.iloc[i,0]/df_main.iloc[i-6,0]-1)*100
    df_main['Alpha3'] = alpha3.values()
    alpha4 = {}
    for i in range(len(df_main)):
        if i <= 4:
            alpha4[i] = 0
        else:
            alpha4[i] = np.mean(df_main.iloc[i-5:i+1,0])/df_main.iloc[i,0]
    df_main['Alpha4'] = alpha4.values()
    df_main['Alpha5'] = np.sign(df_main['volume']-df_main['volume'].shift(1))*(
        -1*(df_main['last']-df_main['last'].shift(1)))
    df_main['Alpha6'] = calculateMACD(df_main['last'])[0]
    df_main['Alpha7'] = calculateMACD(df_main['last'])[1]
    df_main['Alpha8'] = calculateMACD(df_main['last'])[2]
    df_main['Alpha9'] = RSI(df_main['last'])
    df_main['Alpha10'] = getPSY(np.array(df_main['last']))
    df_main['Alpha11'] = (df_main['last']/df_main['last'].rolling(6,min_periods = 1).mean()-1)*100
    alpha12 = {}
    for i in range(len(df_main)):
        if i <= 19:
            alpha12[i] = 0
        else:
            alpha12[i] = -(df_main.iloc[i,0]/df_main.iloc[i-20,0]-1)
    df_main['Alpha12'] = alpha12.values()
    alpha13 = {}
    for i in range(len(df_main)):
        if i == 0:
            alpha13[i] = 0
        else:
            if df_main.iloc[i,0] <= df_main.iloc[i-1,0]:
                a = df_main.iloc[i,0] / df_main.iloc[i-1,0]-1
            elif df_main.iloc[i,0] == df_main.iloc[i-1,0]:
                a = 0
            else:
                a = 1-df_main.iloc[i-1,0]/df_main.iloc[i,0]
            alpha13[i] = a
    df_main['Alpha13'] = alpha13.values()
    M = {}
    A = {}
    for i in range(len(df_main)):
        if i ==0:
            M[i] = 0
            A[i] = 0
        else:
            M[i] = np.max(df_main.iloc[i,1]-df_main.iloc[i-1,1],0)
            A[i] = np.abs(df_main.iloc[i,1]-df_main.iloc[i-1,1])
    df_main['M'] = M.values()
    df_main['A'] = A.values()
    df_main['Alpha14'] = df_main['M'].rolling(6).mean()/df_main['A'].rolling(6).mean()*100
    del df_main['M']
    del df_main['A']
    df_main['pct_chg'] = (df_main['last'].shift(-1)/df_main['last']-1)*100
    df_train = df_main['2013-01-04':'2019-12-30']
    df_test = df_main['2020-01-05':'2021-03-19']
    if len(df_test)<50 or len(df_train)<100:
        continue
    Dict[stock] = df_main
    Dict[stock].to_csv('./train_stock/'+name)
                
    
    
    