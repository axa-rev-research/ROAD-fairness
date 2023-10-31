
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    if y_z_0.mean() !=0:
        odds = y_z_1.mean() / y_z_0.mean()
    if y_z_0.mean() ==0:
        odds = -1 
    return np.min([odds, 1/odds]) * 100

def DI(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = np.abs(y_z_1.mean() - y_z_0.mean())
    return odds

def DispFNR(y_pred, y, z_values, threshold=0.5):
    ypred_z_1 = y_pred > threshold if threshold else y_pred[z_values == 1]
    ypred_z_0 = y_pred > threshold if threshold else y_pred[z_values == 0]
    result=abs(ypred_z_1[(y==1) & (z_values==0)].mean()-ypred_z_1[(y==1) & (z_values==1)].mean())
    return result
def DispFPR(y_pred, y, z_values, threshold=0.5):
    ypred_z_1 = y_pred > threshold if threshold else y_pred[z_values == 1]
    ypred_z_0 = y_pred > threshold if threshold else y_pred[z_values == 0]
    result=abs(ypred_z_1[(y==0) & (z_values==0)].mean()-ypred_z_1[(y==0) & (z_values==1)].mean())
    return result

        
def get_subroups_results_OLD(X, y, S, y_pred, 
                          continuous_names = ["age"], 
                          to_exclude=[],
                          to_keep=[],
                          n_min=100, n_q=4):
    """
    Input: df is the pandas dataframe to create bins in
    Out: pd DataFrame segmented, with local accuracy and DI
    """

    df_results = X.copy()
    
    if len(to_exclude) > 0:
        for c in to_exclude:
            del df_results[c]
    
    if len(to_keep) > 0:
        df_results = df_results[to_keep]
    
    subgroup_cols = df_results.columns
    
    # treat continuous: make quantiles
    for c in continuous_names:
        df_results[c+"_q"] = pd.qcut(df_in[c], q=n_q)#(df_results[c]>AGE_LIMIT).astype('int') #TODO
        del df_results[c]
        subgroup_cols.append(c+"_q")
    
    df_results["Yhat"] = y_pred
    df_results["ytrue"] = y
    df_results["sensitive"] = S
    df_results["acc"] = (df_results["Yhat"]==df_results["ytrue"])
    df_results["acc2"] = (df_results["Yhat"]==df_results["ytrue"])

    # create subgroup variable
    df_grouped = df_results.groupby([c for c in subgroup_cols])
    df_results["group_label"] = df_grouped.grouper.group_info[0]
    
    #g = df_results.groupby([c for c in df_results.columns if c not in ['ytrue', 'Yhat', "acc", "acc2"]]) #create segments (using s)
    g = df_results.groupby(["group_label", "sensitive"])
    
    df_results2 = g.agg({'ytrue':'count', 'Yhat':'mean',  'acc':'mean', 'acc2':'sum'}) # size(seg*S), E(^Y|S,X=seg), Acc(f) sur Seg*S 
    df_results2["acc"] = df_results2["acc"].round(3)
    df3 = df_results2.reset_index().groupby("group_label").agg({"ytrue":'sum',"Yhat":lambda x: x.max() - x.min(), "acc": lambda x: x.tolist(), "acc2":'sum'}) #aggregate genders and get: size(seg), |E(^Y|S=1)-E(^Y|S=0)|, [acc(f|seg0), acc(f|seg1)

    
    df3 = df3[df3["ytrue"]>=n_min].reset_index().rename(columns={"ytrue": "n_obs", "Yhat": "Local DI", "acc":"Acc by S", "acc2":"Local Acc"}) #filter by size of segments and rename columns
    df3["Local Acc"] = df3["Local Acc"] / df3["n_obs"]
    df3["Local Acc"] = df3["Local Acc"].round(3)
    df3["Local DI"]= df3["Local DI"].round(3) # arrondi 3 chiffres
    return df3




        
def get_subroups(X, S, 
                          continuous_names = ["age"], 
                          how_continuous='quantile',
                          to_exclude=[],
                          to_keep=[],
                          n_min=100, n_q=4):
    """
    Input: df is the pandas dataframe to create bins in
    Out: pd DataFrame segmented, with local accuracy and DI
    """

    df_results = X.copy()
    
    if len(to_exclude) > 0:
        for c in to_exclude:
            del df_results[c]
    
    if len(to_keep) > 0:
        df_results = df_results[to_keep]
    
    subgroup_cols = list(df_results.columns)
    
    # treat continuous: make quantiles
    for c in continuous_names:
        if how_continuous == 'quantile':
            df_results[c+"_q"] = pd.qcut(df_results[c], q=n_q) #TODO
        elif how_continuous == 'bins':
            df_results[c+"_q"] = pd.cut(df_results[c], bins=[k*n_q for k in range(int(df_results[c].max()/n_q)+2)], 
                                       ordered=True)
        else: 
            raise ValueError('Parameter how_continous ùmust be either "quantile" or "bins". ')
        del df_results[c]
        subgroup_cols = list(set(subgroup_cols) - set([c]))
        subgroup_cols.append(c+"_q")
    
    df_results["sensitive"] = S

    # create subgroup variable
    df_grouped = df_results.groupby([c for c in subgroup_cols])
    df_results["group_label"] = df_grouped.grouper.group_info[0]
    
    group_description = df_results.groupby('group_label').agg(lambda x: set(x))
    del group_description["sensitive"]
    
    
    
    return df_results, df_grouped, group_description



def get_subgroups_random(X, S, n_groups=4):
    
    df_results = X.copy()
    df_results["sensitive"] = S
    
    n_partition = int(df_results.shape[0]/n_groups)
    print("Partitionning into %i subgroups. Subgroup size will be %i"%(n_groups, n_partition))
    index = list(df_results.index)
    np.random.shuffle(index)
    df_results_s = df_results.loc[index]
    
    group_label = np.array([[i] * n_partition for i in range(n_groups)]).flatten()
    group_label = np.append(group_label, [0] * (df_results.shape[0] - group_label.shape[0]))
    
    df_results_s["group_label"] = group_label
    
    df_out = df_results_s.loc[df_results.index]
    
    return df_out, -1 , -1 
        
        
def get_subroups_results_EO(df_groups, y, S, y_pred, 
                          n_min=100):
    """
    Input: df is the pandas dataframe to create bins in
    Out: pd DataFrame segmented, with local accuracy and DI
    """

    df_results = df_groups.copy()
    
    df_results["Yhat"] = y_pred
    df_results["ytrue"] = y
    df_results["sensitive"] = S
    df_results["acc"] = (df_results["Yhat"]==df_results["ytrue"])
    df_results["acc2"] = (df_results["Yhat"]==df_results["ytrue"])

    # create subgroup variable    
    g = df_results.groupby(["group_label", "sensitive"])
    
    df_results2 = g.agg({'ytrue':'count', 'Yhat':'mean',  'acc':'mean', 'acc2':'sum'}) # size(seg*S), E(^Y|S,X=seg), Acc(f) sur Seg*S 
    df_results2["acc"] = df_results2["acc"].round(3)
    df3 = df_results2.reset_index().groupby("group_label").agg({"ytrue":'sum',"Yhat":lambda x: x.max() - x.min(), "acc": lambda x: x.tolist(), "acc2":'sum'}) #aggregate genders and get: size(seg), |E(^Y|S=1)-E(^Y|S=0)|, [acc(f|seg0), acc(f|seg1)

    
    df3 = df3[df3["ytrue"]>=n_min].reset_index().rename(columns={"ytrue": "n_obs", "Yhat": "Local DI", "acc":"Acc by S", "acc2":"Local Acc"}) #filter by size of segments and rename columns
    df3["Local Acc"] = df3["Local Acc"] / df3["n_obs"]
    df3["Local Acc"] = df3["Local Acc"].round(3)
    df3["Local DI"]= df3["Local DI"].round(3) # arrondi 3 chiffres

    return df3

        
def get_subroups_results(df_groups, y, S, y_pred, 
                          n_min=100):
    """
    Input: df is the pandas dataframe to create bins in
    Out: pd DataFrame segmented, with local accuracy and DI
    """

    df_results = df_groups.copy()
    
    df_results["Yhat"] = y_pred
    df_results["ytrue"] = y
    df_results["sensitive"] = S
    df_results["acc"] = (df_results["Yhat"]==df_results["ytrue"])
    df_results["acc2"] = (df_results["Yhat"]==df_results["ytrue"])

    # create subgroup variable    
    g = df_results.groupby(["group_label", "sensitive"])
    
    df_results2 = g.agg({'ytrue':'count', 'Yhat':'mean',  'acc':'mean', 'acc2':'sum'}) # size(seg*S), E(^Y|S,X=seg), Acc(f) sur Seg*S 
    df_results2["acc"] = df_results2["acc"].round(3)
    df3 = df_results2.reset_index().groupby("group_label").agg({"ytrue":'sum',"Yhat":lambda x: x.max() - x.min(), "acc": lambda x: x.tolist(), "acc2":'sum'}) #aggregate genders and get: size(seg), |E(^Y|S=1)-E(^Y|S=0)|, [acc(f|seg0), acc(f|seg1)

    
    df3 = df3[df3["ytrue"]>=n_min].reset_index().rename(columns={"ytrue": "n_obs", "Yhat": "Local DI", "acc":"Acc by S", "acc2":"Local Acc"}) #filter by size of segments and rename columns
    df3["Local Acc"] = df3["Local Acc"] / df3["n_obs"]
    df3["Local Acc"] = df3["Local Acc"].round(3)
    df3["Local DI"]= df3["Local DI"].round(3) # arrondi 3 chiffres
    return df3


def get_subroups_results_EO(df_groups, y, S, y_pred, 
                          n_min=100):
    """
    Input: df is the pandas dataframe to create bins in
    Out: pd DataFrame segmented, with local accuracy and DI
    """

    df_results = df_groups.copy()
    
    df_results["Yhat"] = y_pred
    df_results["ytrue"] = y
    df_results["sensitive"] = S
    df_results["acc"] = (df_results["Yhat"]==df_results["ytrue"])
    df_results["acc2"] = (df_results["Yhat"]==df_results["ytrue"])
    df_results["Yhat2"] = y_pred
    df_results["Yhat3"] = y_pred

    # create subgroup variable    
    g = df_results.groupby(["group_label", "sensitive"])
    df_results2 = g.agg({'ytrue':'count', 'Yhat':'mean',  'acc':'mean', 'acc2':'sum'}) # size(seg*S), E(^Y|S,X=seg), Acc(f) sur Seg*S 
    df_results2["acc"] = df_results2["acc"].round(3)
    df3 = df_results2.reset_index().groupby("group_label").agg({"ytrue":'sum',"Yhat":lambda x: x.max() - x.min(), "acc": lambda x: x.tolist(), "acc2":'sum'}) #aggregate genders and get: size(seg), |E(^Y|S=1)-E(^Y|S=0)|, [acc(f|seg0), acc(f|seg1)

    # create subgroup variable
    g = df_results.groupby(["group_label", "sensitive","ytrue"])
    df_results2 = g.agg({'Yhat2':'mean','Yhat3':'mean'}) # size(seg*S), E(^Y|S,X=seg), Acc(f) sur Seg*S 

    df_results2=df_results2.reset_index()
    dfFN = df_results2.reset_index().where(df_results2.ytrue == 1).groupby(["group_label"]).agg({"Yhat2":lambda x: x.max() - x.min()}).reset_index()
    dfFP = df_results2.reset_index().where(df_results2.ytrue == 0).groupby(["group_label"]).agg({"Yhat3":lambda x: x.max() - x.min()}).reset_index()

    df3l = df3.merge(dfFN, on='group_label', how='left').merge(dfFP, on='group_label', how='left')

    df3l = df3l[df3l["ytrue"]>=n_min].reset_index().rename(columns={"ytrue": "n_obs", "Yhat": "Local DI", "acc":"Acc by S", "acc2":"Local Acc","Yhat2":"Local FNR","Yhat3":"Local FPR"}) #filter by size of segments and rename columns
    #print(df3l)

    df3l["Local Acc"] = df3l["Local Acc"] / df3l["n_obs"]
    df3l["Local Acc"] = df3l["Local Acc"].round(3)
    df3l["Local DI"]= df3l["Local DI"].round(3) # arrondi 3 chiffres

    return df3l


def get_scores_without_agg(df_groups, X, y, S, y_pred, n_min, topK_DI=[1, 3], q_DI=4, topK_acc=[1], q_acc=4):

    sr = get_subroups_results(df_groups, y, S, y_pred, n_min)
 
    out_dict = {}
    
    out_dict["Global Acc"] = accuracy_score(y, y_pred)
    out_dict["Global DI"] = DI(y_pred, S)
    

    for k in topK_DI:
        topk = sr.sort_values("Local DI", ascending=False)["Local DI"][:k].mean()
        out_dict["top"+str(k)+"_DI"] = topk

    for k in topK_acc:
        topk = sr.sort_values("Local Acc", ascending=True)["Local Acc"][:k].mean()
        out_dict["worst"+str(k)+"_acc"] = topk

    for q in range(q_DI):
        step = 1.0 / q_DI

        val_q = np.quantile(sr["Local DI"], q=[q*step])
        out_dict["q_DI_" + str(q*step)] = float(val_q)
        
    out_dict["Local DI"] = sr["Local DI"].values

    # global acc
    # global DI

    return out_dict 


def get_scores(df_groups, X, y, S, y_pred, n_min, topK_DI=[1, 3], q_DI=4, topK_acc=[1], q_acc=4):

    sr = get_subroups_results(df_groups, y, S, y_pred, n_min)
    print(sr)
    out_dict = {}
    
    out_dict["Global Acc"] = accuracy_score(y, y_pred)
    out_dict["Global DI"] = DI(y_pred, S)
    

    for k in topK_DI:
        topk = sr.sort_values("Local DI", ascending=False)["Local DI"][:k].mean()
        out_dict["top"+str(k)+"_DI"] = topk

    for k in topK_acc:
        topk = sr.sort_values("Local Acc", ascending=True)["Local Acc"][:k].mean()
        out_dict["worst"+str(k)+"_acc"] = topk

    for q in range(q_DI):
        step = 1.0 / q_DI

        val_q = np.quantile(sr["Local DI"], q=[q*step])
        out_dict["q_DI_" + str(q*step)] = float(val_q)
        
    out_dict["Local DI"] = sr["Local DI"].values

    # global acc
    # global DI

    return out_dict 


def get_scores_EO(df_groups, X, y, S, y_pred, n_min, topK_DI=[1, 3], q_DI=4,  topK_EO=[1, 3],q_EO=4, topK_acc=[1], q_acc=4):

    sr = get_subroups_results_EO(df_groups, y, S, y_pred, n_min)

    out_dict = {}
    
    out_dict["Global Acc"] = accuracy_score(y, y_pred)
    out_dict["Global DI"] = DI(y_pred, S)
    out_dict["Global FNR"] = DispFNR(y_pred, y, S)
    out_dict["Global FPR"] = DispFPR(y_pred, y, S)


    for k in topK_DI:
        topk = sr.sort_values("Local DI", ascending=False)["Local DI"][:k].mean()
        out_dict["top"+str(k)+"_DI"] = topk

    for k in topK_EO:
        topk = sr.sort_values("Local FNR", ascending=False)["Local FNR"][:k].mean()
        out_dict["top"+str(k)+"_FNR"] = topk
        topk = sr.sort_values("Local FPR", ascending=False)["Local FPR"][:k].mean()
        out_dict["top"+str(k)+"_FNR"] = topk
        
    for k in topK_acc:
        topk = sr.sort_values("Local Acc", ascending=True)["Local Acc"][:k].mean()
        out_dict["worst"+str(k)+"_acc"] = topk

    for q in range(q_DI):
        step = 1.0 / q_DI

        val_q = np.quantile(sr["Local DI"], q=[q*step])
        out_dict["q_DI_" + str(q*step)] = float(val_q)

    for q in range(q_EO):
        step = 1.0 / q_EO

        val_q = np.quantile(sr["Local FNR"], q=[q*step])
        out_dict["q_FNR_" + str(q*step)] = float(val_q)  

        val_q = np.quantile(sr["Local FPR"], q=[q*step])
        out_dict["q_FPR_" + str(q*step)] = float(val_q)  
        
        
    out_dict["Local DI"] = sr["Local DI"].values
    out_dict["Local FNR"] = sr["Local FNR"].values
    out_dict["Local FPR"] = sr["Local FPR"].values

    # global acc
    # global DI

    return out_dict 

# extracting Pareto Front - takes a bit of time when used
def pareto_df(dataset, f1, f2):
    dff = dataset[[f1, f2]].sort_values(f1)
    dff["pareto"] = np.ones(dff.shape[0])
    for i in range(len(dff)):
        if i%100 == 0:
            if i > 1500:
                print(i)
        for j in range(len(dff)):
            cond1 = (dff[f1].iloc[j] < dff[f1].iloc[i]) and (dff[f2].iloc[j] >= dff[f2].iloc[i])
            cond2 = (dff[f1].iloc[j] <= dff[f1].iloc[i]) and (dff[f2].iloc[j] > dff[f2].iloc[i])
            if cond1 or cond2: # si tous les points sont soient plus unfair. Soit
                dff['pareto'].iloc[i] = 0
    return dff

