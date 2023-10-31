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

from models.Road_EO import ROAD_EO
from models.Broad import BROAD

from data.load_fairness_data import LOAD_DATA_TRAIN_TEST
from utils.fairness_utils import get_subroups_results, get_scores_EO, get_subroups, get_subgroups_random

# In[4]:


X_train0, X_test0,X_train, X_test, y_train, y_test, S_train, S_test, column_names = LOAD_DATA_TRAIN_TEST('adult')

X_train.shape, X_test.shape


# In[5]:


#get_ipython().run_line_magic('matplotlib', 'inline')

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
        self.fc1_s = nn.Linear(2, 64)
        self.fc2_s = nn.Linear(64, 32)
        self.fc3_s = nn.Linear(32, 16)
        self.fc4_s = nn.Linear(16, 1)

    def forward(self, x,y):
        combined = torch.cat((x.view(x.size(0), -1),
                      y.view(y.size(0), -1)), dim=1)
        x = self.fc1_s(combined)
        x = torch.relu(x)
        x = self.fc2_s(x)
        x = torch.relu(x)
        x = self.fc3_s(x)
        x = torch.relu(x)
        x = self.fc4_s(x)
        return x




taille_z = 64


# ## DRO Zhang

# In[7]:


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
        self.fc1_s = nn.Linear(2, 64)
        self.fc2_s = nn.Linear(64, 32)
        self.fc3_s = nn.Linear(32, 16)
        self.fc4_s = nn.Linear(16, 1)

    def forward(self, x,y):
        combined = torch.cat((x.view(x.size(0), -1),
                      y.view(y.size(0), -1)), dim=1)
        x = self.fc1_s(combined)
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
        self.fc1 = nn.Linear(X_train.shape[1]+2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x,y):
        combined = torch.cat((x.view(x.size(0), -1),
                      y.view(y.size(0), -1)), dim=1)
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.exp(self.fc3(x))
        return x 




# In[8]:

### MULTIPLE RUNS

import warnings
warnings.filterwarnings("ignore")


scores = []
scores_rdm = []


scores2014 = []
scores_rdm2014 = []

scores2015 = []
scores_rdm2015 = []

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



#gr = get_subgroups_random(X_test0, S_test, n_groups=4)

X_train0,X_test0, X_test20140, X_test20150,X_train, X_test, X_test2014, X_test2015, y_train, y_test, y_test2014, y_test2015, S_train, S_test, S_test2014, S_test2015, column_names = LOAD_DATA_TRAIN_TEST('adult_drift')




gr = get_subroups(X_test0, S_test, continuous_names=["age"], how_continuous='bins', n_min=N_MIN_GROUP, n_q=10,
         to_keep=["age"]+["race"]+[c for c in X_test0.columns if "country" in c])

gr2014 = get_subroups(X_test20140, S_test2014, continuous_names=["age"], how_continuous='bins', n_min=N_MIN_GROUP, n_q=10,to_keep=["age"]+["race"]+[c for c in X_test20140.columns if "country" in c])

gr2015 = get_subroups(X_test20150, S_test2015, continuous_names=["age"], how_continuous='bins', n_min=N_MIN_GROUP, n_q=10,to_keep=["age"]+["race"]+[c for c in X_test20150.columns if "country" in c])

print(X_test20140.shape)
#X_test2014= X_test20140.values
for i in range(N_RUNS):
    for tau in TAUS:
        for lambda_ in LAMBDAS:
            print('======================== ITERATION %i - LAMBDA %f - TAU %f'%(i,lambda_,tau))
            #X_train0, X_test0,X_train, X_test, y_train, y_test, S_train, S_test, column_names = LOAD_DATA_TRAIN_TEST('adult')
            #gr = get_subroups(X_test0, S_test, continuous_names=["age"], how_continuous='bins', n_min=N_MIN_GROUP, n_q=10,
            #                     to_keep=["age"]+["race"])#[c for c in X_train0.columns if "c_charge_desc" in c])
            #X_train.shape, X_test2014.shape, X_test2015.shape

            if sys.argv[4] == "ROAD":
                model = ROAD_EO(learning_rate=0.0001,batch_size= 512,
                            lamb= lambda_, lambda_r=tau, num_epochs=100,
                            NN_y= NN_y_dro, NN_s= NN_s_dro, NN_a=NN_a_dro, GPU=str(sys.argv[3]))
                
            if sys.argv[4] == "BROAD":
                model = BROAD(learning_rate=0.0001,batch_size= 512,
                            lamb= lambda_, lambda_r=tau, num_epochs=200,
                            NN_y= NN_y_dro, NN_s= NN_s_dro, NN_a=NN_a_dro, GPU=str(sys.argv[3]))

                
            if sys.argv[4][:4] == "ROAD":  #[:19]
                print('Number of iterations of S is: ', nbS)
                print(' Number of iterations of R is: ', nbR)
                print(S_train.shape)
                model.fit(X_train, y_train, S_train, nb_iter_s = nbS, nb_iter_r = nbR)
            else:
                y_pred_dro = model.fit(X_train, y_train, S_train)

            y_pred_dro = model.predict(X_test)
            
            dro_scores = get_scores_EO(gr[0], X_test0, y_test, S_test, y_pred_dro, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores["model"] = str(sys.argv[4])
            dro_scores["run_id"] = i
            dro_scores["lambda"] = lambda_
            dro_scores["tau"] = tau
            scores.append(dro_scores)
            
            gr_rdm = get_subgroups_random(X_test0, S_test, n_groups=20)
            dro_scores_rdm = get_scores_EO(gr_rdm[0], X_test0, y_test, S_test, y_pred_dro, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores_rdm["model"] = str(sys.argv[4])
            dro_scores_rdm["run_id"] = i
            dro_scores_rdm["lambda"] = lambda_
            dro_scores_rdm["tau"] = tau

            scores_rdm.append(dro_scores_rdm)
            
            ############ 2014 ############
            print(X_test20140)

            y_pred_dro2014 = model.predict(X_test2014)
            
            print(y_test2014.shape)
            print(X_test20140.shape)
            print(y_pred_dro2014.shape)
            dro_scores2014 = get_scores_EO(gr2014[0], X_test20140, y_test2014, S_test2014, y_pred_dro2014, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            
            print(dro_scores2014)

            dro_scores2014["model"] = str(sys.argv[4])
            dro_scores2014["run_id"] = i
            dro_scores2014["lambda"] = lambda_
            dro_scores2014["tau"] = tau
            scores2014.append(dro_scores2014)
            
            gr_rdm2014 = get_subgroups_random(X_test20140, S_test2014, n_groups=20)
            dro_scores_rdm2014 = get_scores_EO(gr_rdm2014[0], X_test20140, y_test2014, S_test2014, y_pred_dro2014, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores_rdm2014["model"] = str(sys.argv[4])
            dro_scores_rdm2014["run_id"] = i
            dro_scores_rdm2014["lambda"] = lambda_
            dro_scores_rdm2014["tau"] = tau
            
            scores2014.append(dro_scores2014)
            scores_rdm2014.append(dro_scores_rdm2014)
            
            ############ 2015 ############

            y_pred_dro2015 = model.predict(X_test2015)
            dro_scores2015 = get_scores_EO(gr2015[0], X_test20150, y_test2015, S_test2015, y_pred_dro2015, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores2015["model"] = str(sys.argv[4])
            dro_scores2015["run_id"] = i
            dro_scores2015["lambda"] = lambda_
            dro_scores2015["tau"] = tau
            scores2015.append(dro_scores2015)
            
            
            gr_rdm2015 = get_subgroups_random(X_test20150, S_test2015, n_groups=20)
            dro_scores_rdm2015 = get_scores_EO(gr_rdm2015[0], X_test20150, y_test2015, S_test2015, y_pred_dro2015, n_min=N_MIN_GROUP,
                          topK_DI=[1, 3], q_DI=10, topK_acc=[1], q_acc=4)
            dro_scores_rdm2015["model"] = str(sys.argv[4])
            dro_scores_rdm2015["run_id"] = i
            dro_scores_rdm2015["lambda"] = lambda_
            dro_scores_rdm2015["tau"] = tau
            
            scores2015.append(dro_scores2015)
            scores_rdm2015.append(dro_scores_rdm2015)
                        
            
            
            
            df_out = pd.DataFrame(scores)
            df_out.to_csv("./../results_Adultdrifted/" + sys.argv[5] + "Shifted_EO.csv")
            df_out_rdm = pd.DataFrame(scores_rdm)
            df_out_rdm.to_csv("./../results_Adultdrifted/" + sys.argv[5] + "Shifted_EO_rdm.csv")

            df_out2014 = pd.DataFrame(scores2014)
            df_out2014.to_csv("./../results_Adultdrifted/" + sys.argv[5] + "2014Shifted_EO.csv")
            df_out_rdm2014 = pd.DataFrame(scores_rdm2014)
            df_out_rdm2014.to_csv("./../results_Adultdrifted/" + sys.argv[5] + "2014Shifted_EO_rdm.csv")

            df_out2015 = pd.DataFrame(scores2015)
            df_out2015.to_csv("./../results_Adultdrifted/" + sys.argv[5] + "2015Shifted_EO.csv")
            df_out_rdm2015 = pd.DataFrame(scores_rdm2015)
            df_out_rdm2015.to_csv("./../results_Adultdrifted/" + sys.argv[5] + "2015Shifted_EO_rdm.csv")
# In[ ]:


### MULTIPLE RUNS



