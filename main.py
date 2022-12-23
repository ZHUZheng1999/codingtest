# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Model import MLP
seq = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    mse = []
    folder_name = os.listdir('./train_stock')
    model = MLP(N_Inputs=16*seq,classes=1)
    model = model.to(device)
    
    optimiser = torch.optim.Adam(model.parameters(),lr = 0.0003)
    loss_fn = torch.nn.MSELoss(size_average = True)
    for turn,name in enumerate(folder_name):
        stock = name[:-4]
        df_main = pd.read_csv('./train_stock/'+name)
        df_main.set_index('date',drop=True,inplace=True)
        df_main.dropna(inplace = True)
        
        data_feat, data_target = [],[]
        for index in range(len(df_main)-seq):
            data_normal = df_main.iloc[index:index+seq,:-1]
            for colname in data_normal.columns:
                data_normal[colname] = (data_normal[colname]-np.mean(data_normal[colname]))/(np.std(data_normal[colname])+0.000001)
            data_feat.append(data_normal.values)
            data_target.append(df_main['pct_chg'][index+seq-1])
            
        data_feat = np.array(data_feat)
        data_target = np.array(data_target)
        
        train_size = len(df_main['2013-01-04':'2019-12-30'])-seq
        test_size = len(df_main['2020-01-06':'2021-03-18'])
        
        trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,16*seq)).type(torch.Tensor)
        testX = torch.from_numpy(data_feat[train_size:train_size+test_size].reshape(-1,16*seq)).type(torch.Tensor)
        
        trainY = torch.from_numpy(data_target[:train_size].reshape(-1,1)).type(torch.Tensor)
        testY = torch.from_numpy(data_target[train_size:train_size+test_size].reshape(-1,1)).type(torch.Tensor)
        num_epochs = 200
        
        print(name)
        
        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            trainX = trainX.to(device)
            y_train_pred = model(trainX)
            trainY = trainY.to(device)
            loss = loss_fn(y_train_pred,trainY)
            if (t+1)%100 ==0 and t!=0:
                print("Epoch",t,"MSE:",loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            
        with torch.no_grad():
            model.eval()
            testX = testX.to(device)
            y_test_pred = model(testX)
            testY = testY.to(device)
            print("MSE in the test set is:",loss_fn(y_test_pred,testY).item())
            
        test_pred_value = y_test_pred.cpu().detach().numpy()
        test_true_value = testY.cpu().detach().numpy()
        
        
        
        from sklearn.metrics import mean_squared_error
        
        mse.append(mean_squared_error(test_pred_value, test_true_value))
        print("mse: ",mse)
        df_pred = df_main['2020-01-06':'2021-03-19']
        df_pred.insert(loc=len(df_pred.columns),column = 'pred', value = test_pred_value)
        df_pred.to_csv('./predict/'+name)
        torch.save(model.state_dict(),'./Modelsave/model_period_{}.pth'.format(turn))
        
    print(np.mean(mse))
        
        