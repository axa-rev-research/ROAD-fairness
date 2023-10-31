#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets as ds


# In[2]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from models.Road import ROAD
from models.Broad import BROAD

from data.load_fairness_data import LOAD_DATA_TRAIN_TEST
from utils.fairness_utils import get_subroups_results, get_scores, get_subroups, get_subgroups_random


X_train0, X_test0,X_train, X_test, y_train, y_test, S_train, S_test, column_names = LOAD_DATA_TRAIN_TEST('compass')

X_train.shape, X_test.shape


from torch import nn
import torch

    
class NN_y(nn.Module):
    def __init__(self):
        super(NN_y, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
            

class NN_s(nn.Module):
    def __init__(self):
        super(NN_s, self).__init__()
        self.fc1_s = nn.Linear(1, 64)
        self.fc2_s = nn.Linear(64, 32)
        self.fc3_s = nn.Linear(32, 16)
        self.fc4_s = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1_s(x)
        x = torch.relu(x)
        x = self.fc2_s(x)
        x = torch.relu(x)
        x = self.fc3_s(x)
        x = torch.relu(x)
        x = self.fc4_s(x)
        return x



taille_z = 64



class NN_y_dro(nn.Module):
    def __init__(self):
        super(NN_y_dro, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
            

class NN_s_dro(nn.Module):
    def __init__(self):
        super(NN_s_dro, self).__init__()
        self.fc1_s = nn.Linear(1, 64)
        self.fc2_s = nn.Linear(64, 32)
        self.fc3_s = nn.Linear(32, 16)
        self.fc4_s = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1_s(x)
        x = torch.relu(x)
        x = self.fc2_s(x)
        x = torch.relu(x)
        x = self.fc3_s(x)
        x = torch.relu(x)
        x = self.fc4_s(x)
        return x


class NN_a_dro(nn.Module):
    def __init__(self):
        super(NN_a_dro, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1]+1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.exp(self.fc3(x))
        return x 




# In[8]:


# # loop results

# In[15]:





# In[ ]:
#LAMBDAS = 
# In[ ]:


### MULTIPLE RUNS

import warnings
warnings.filterwarnings("ignore")


scores = []
scores_rdm = []

N_RUNS = 2




try:
    nb = int(sys.argv[6])
except IndexError:
    nb = 20
print('Lambda from ', 0, 'to ', float(sys.argv[1]))
print('Number of iterations of lambda', nb)


try:
    nb2 = int(sys.argv[7])
except IndexError:
    nb2 = 10
print(' Tau from ', 0, 'to ', float(sys.argv[2]))
print('Number of iterations of tau', nb2)

LAMBDAS = np.linspace(0.0, float(sys.argv[1]), num=nb)
TAUS = np.linspace(0.0, float(sys.argv[2]), num=nb2)
if float(sys.argv[2])==0:
    TAUS = [0]
N_MIN_GROUP = 50
N_EPOCHS = 200
#N_MIN_GROUPS = [10, 20, 30, 50, 80, 100, 200]

import warnings
warnings.filterwarnings("ignore")





try:
    nbS = int(sys.argv[8])
except IndexError:
    nbS = 30
try:
    nbR = int(sys.argv[9])
except IndexError:
    nbR = 30




for tau in TAUS:
    for lambda_ in LAMBDAS:
        for i in range(N_RUNS):
            print('======================== ITERATION %i'%i)
            X_train0, X_test0,X_train, X_test, y_train, y_test, S_train, S_test, column_names = LOAD_DATA_TRAIN_TEST('compass')
            gr = get_subroups(X_test0, S_test, continuous_names=["age"], how_continuous='bins', n_min=N_MIN_GROUP, n_q=10,
                         to_keep=["age"]+["sex"])
            print(sys.argv[4])
            #DRO ZHANG
            if sys.argv[4] == "ROAD":
                model = ROAD(learning_rate=0.0001,batch_size= 1024,
                            lamb= lambda_, lambda_r=tau, num_epochs=100,
                            NN_y= NN_y_dro, NN_s= NN_s_dro, NN_a=NN_a_dro, GPU=str(sys.argv[3]))

            if sys.argv[4] == "BROAD":
                model = BROAD(learning_rate=0.0001,batch_size= 1024,
                            lamb= lambda_, lambda_r=tau, num_epochs=175,
                            NN_y= NN_y_dro, NN_s= NN_s_dro, NN_a=NN_a_dro, GPU=str(sys.argv[3]))

                
            if sys.argv[4] == "ROAD":  #[:19]
                print('Number of iterations of S is: ', nbS)
                print(' Number of iterations of R is: ', nbR)
                model.fit(X_train, y_train, S_train, nb_iter_s = nbS, nb_iter_r = nbR)
            else:
                y_pred_dro = model.fit(X_train, y_train, S_train)

            y_pred_dro = model.predict(X_test)
            
            
            dro_scores = get_scores(gr[0], X_test0, y_test, S_test, y_pred_dro, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores["model"] = str(sys.argv[4])
            dro_scores["run_id"] = i
            dro_scores["lambda"] = lambda_
            dro_scores["tau"] = tau
            scores.append(dro_scores)
            
            
            gr_rdm = get_subgroups_random(X_test0, S_test, n_groups=20)
            dro_scores_rdm = get_scores(gr_rdm[0], X_test0, y_test, S_test, y_pred_dro, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores_rdm["model"] = str(sys.argv[4])
            dro_scores_rdm["run_id"] = i
            dro_scores_rdm["lambda"] = lambda_
            dro_scores_rdm["tau"] = tau
            
            scores.append(dro_scores)
            scores_rdm.append(dro_scores_rdm)
 
            
            df_out = pd.DataFrame(scores)
            df_out.to_csv("./../results_compass/" + sys.argv[5] + "_DP.csv")
            df_out_rdm = pd.DataFrame(scores_rdm)
            df_out_rdm.to_csv("./../results_compass/" + sys.argv[5] + "_DP_rdm.csv")

# In[ ]:


### MULTIPLE RUNS



