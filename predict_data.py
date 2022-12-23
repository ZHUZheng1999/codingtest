# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
import os

data_raw = pd.read_csv('./train_stock/1332 JT.csv')
predict_df = pd.DataFrame()
predict_df['date'] = data_raw['date']
list_data = [0 for i in range(len(predict_df['date']))]
predict_df.set_index('date',drop = True,inplace = True)
true_df = pd.DataFrame()
columns = []
folder_name = os.listdir('./predict')
index = 0
for name in folder_name:
    stock = name[:-4]
    predict_df.insert(loc=index,column = stock, value = list_data)
    index+=1
true_df = predict_df
for name in folder_name:
    stock = name[:-4]
    df = pd.read_csv('./predict/'+name,index_col='date')
    for i,index in enumerate(df.index):
        predict_df.loc[index,stock] = df.iloc[i,-1]

for name in folder_name:
    stock = name[:-4]
    df = pd.read_csv('./predict/'+name,index_col='date')
    df['return'] = (df['last']/df['last'].shift(1)-1)*100
    for i,index in enumerate(df.index):
        true_df.loc[index,stock] = df.iloc[i,-1]
predict_df = predict_df['2020-01-07':'2021-03-18']
predict_df.to_csv('./predict_data_all.csv')
true_df = true_df['2020-01-07':'2021-03-18']
true_df.to_csv('./true_df.csv')



    