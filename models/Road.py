

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import random

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader




#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.fairness_utils import p_rule, DI




class ROAD(object):

    def __init__(self, learning_rate, batch_size, lamb, lambda_r,
                 num_epochs, NN_y, NN_s, NN_a, GPU, name="01"):
        self.lambda_ADV = lamb
        self.lambda_r = lambda_r
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.device = torch.device(GPU if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.m_NN_y = NN_y().to(self.device)
        self.m_NN_s = NN_s().to(self.device)
        self.m_NN_a = NN_a().to(self.device)
        self.name = "ROAD_" + str(lamb) + "_" + str(lambda_r) + "_" + name

        self.GPU = GPU

    def fit(self, X_train, y_train, S_train, X_test=None, y_test=None, S_test=None, plot_losses=False, nb_iter_s = 50, nb_iter_r = 50):
        
        batch_no = int(len(X_train) // self.batch_size)+1

        self.optimizer_y = torch.optim.Adam(self.m_NN_y.parameters(), lr=self.learning_rate)
        self.optimizer_s = torch.optim.Adam(self.m_NN_s.parameters(), lr=self.learning_rate)
        self.optimizer_a = torch.optim.Adam(self.m_NN_a.parameters(), lr=self.learning_rate)

        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        criterion2 = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        loss_all, loss_sall, loss_yall = [], [], []
        
        for epoch in range(self.num_epochs):
            x_train, ytrain, strain = shuffle(X_train, np.expand_dims(y_train,axis = 1),np.expand_dims(S_train,axis = 1))
            # Mini batch learning
            epsilon=0.00000000000000001
            for i in range(batch_no):
                start = i * self.batch_size
                end = start + self.batch_size

                x_var = Variable(torch.FloatTensor(x_train[start:end])).to(self.device)
                y_var = Variable(torch.FloatTensor(ytrain[start:end])).to(self.device)
                s_var = Variable(torch.FloatTensor(strain[start:end])).to(self.device)
                x_var_s_var = torch.cat([x_var.view(-1, x_var.shape[1]),s_var.view(-1, 1)],1)
                maskS1 =  torch.nonzero(s_var==1)[:,0]
                maskS0 =  torch.nonzero(s_var==0)[:,0]              
                # Forward + Backward + Optimize
                Ypred_var0 =  self.m_NN_y(x_var).detach()
                
                # Learning S with nb_iter_s iterations
                for l in range(nb_iter_s):
                    self.optimizer_s.zero_grad()
                    Spred_var = self.m_NN_s(Ypred_var0)
                    lossS = torch.mean(criterion2(Spred_var, s_var))
                    lossS.backward()
                    self.optimizer_s.step()

                Spred_var0 = self.m_NN_s(Ypred_var0).detach()

                # Learning Ratio  Apred_var with nb_iter_r iterations
                for m in range(nb_iter_r):
                    self.optimizer_a.zero_grad()
                    Apred_var = self.m_NN_a(x_var_s_var)/torch.mean(self.m_NN_a(x_var_s_var))
                    Apred_var[maskS1] = self.m_NN_a(x_var_s_var)[maskS1]/torch.mean(self.m_NN_a(x_var_s_var)[maskS1])
                    Apred_var[maskS0] = self.m_NN_a(x_var_s_var)[maskS0]/torch.mean(self.m_NN_a(x_var_s_var)[maskS0])

                    lossA = torch.mean(Apred_var * criterion2(Spred_var0, s_var)) + self.lambda_r*torch.mean(Apred_var*torch.log(Apred_var))
                    lossA.backward()
                    self.optimizer_a.step()
                
                # Learning Predictor with only one iteration !!
                self.optimizer_y.zero_grad()
                Ypred_var = self.m_NN_y(x_var)
                #Apred_var = self.m_NN_a(x_var_s_var)/torch.mean(self.m_NN_a(x_var_s_var))
                Apred_var = self.m_NN_a(x_var_s_var)/torch.mean(self.m_NN_a(x_var_s_var))
                Apred_var[maskS1] = self.m_NN_a(x_var_s_var)[maskS1]/torch.mean(self.m_NN_a(x_var_s_var)[maskS1])
                Apred_var[maskS0] = self.m_NN_a(x_var_s_var)[maskS0]/torch.mean(self.m_NN_a(x_var_s_var)[maskS0])
                    
                Spred_var = self.m_NN_s(Ypred_var)
                lossY = criterion(Ypred_var, y_var)

                loss =  - self.lambda_ADV * torch.mean(Apred_var*criterion2(Spred_var, s_var)) + lossY 
                loss.backward()
                self.optimizer_y.step()
                
                loss_sall.append(lossS.item())
                loss_yall.append(lossY.item())
                loss_all.append(loss.item())
                
            if epoch % 5 == 0:
                y_pred2= torch.sigmoid(self.m_NN_y(torch.FloatTensor(X_train).to(self.device))).cpu().data.numpy()
                if X_test is not None:
                    y_pred2t= torch.sigmoid(self.m_NN_y(torch.FloatTensor(X_test).to(self.device))).cpu().data.numpy()
                    print('epoch', epoch, 'loss', loss.cpu().data.numpy(), 'lossS', lossS.cpu().data.numpy(), 'lossY', lossY.cpu().data.numpy(),'P-rule', p_rule(y_pred2,S_train),'ACC_train',accuracy_score(y_train, y_pred2>0.5),
                      'P-ruletest', p_rule(y_pred2t,S_test),'ACC_test',accuracy_score(y_test, y_pred2t>0.5))
                else: 
                    print('epoch', epoch, 'loss', loss.cpu().data.numpy(), 'lossS', lossS.cpu().data.numpy(), 'lossY', lossY.cpu().data.numpy(),'P-rule', p_rule(y_pred2,S_train),'ACC_train',accuracy_score(y_train, y_pred2>0.5))
                    
        if plot_losses:
            df_losses = pd.DataFrame({'loss':loss_all, 'loss_S':loss_sall, 'loss_Y':loss_yall})
            sns.lineplot(data=df_losses)
            plt.show()
            
    def predict(self, X, threshold=0.5):
        return (torch.sigmoid(self.m_NN_y(torch.FloatTensor(X).to(self.device))).cpu().data.numpy()>threshold).astype('int')
    
    def predict_proba(self, X):
        return torch.sigmoid(self.m_NN_y(torch.FloatTensor(X).to(self.device))).cpu().data.numpy()


