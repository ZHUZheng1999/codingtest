# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,N_Inputs,classes):
        super(MLP,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(N_Inputs,512),nn.BatchNorm1d(512),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(512,1024),nn.BatchNorm1d(1024),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(512,classes))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

