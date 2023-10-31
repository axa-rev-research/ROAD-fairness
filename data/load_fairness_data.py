from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def LOAD_DATA_TRAIN_TEST(dname):
    try:
        dparams = DATA_DICT_P[dname]()
        dfunction = DATA_DICT_F[dname]
    except KeyError:
        raise KeyError("Dataset name must be one of the the following: %s"%str(list(DATA_DICT_P.keys())))
    return dfunction(*dparams)

def _get_data_params_adult():
    return [None]


def _get_data_params_german():
    #num = 2
    #sens = 'sex'
    #y = "credit"
    #columns_delete = ['credit','sex-age', "sex"]# 'race','age'']
    #print('Dropping columns:', columns_delete)
    return [None] # num,sens,y,columns_delete

def _get_data_params_lawschool():
    test_size = 0.3
    return [None]

def _get_data_params_compass():
    num = 3
    sens = 'race'
    y = "two_year_recid"
    columns_delete = ['two_year_recid','sex-race', "race"]# 'race','age'']
    print('Dropping columns:', columns_delete)
    return num,sens,y,columns_delete

def _get_data_params_adult_drift():
    size = 0
    #test_year = "2014"
    return [None]

"""def _get_data_params_adult_drift2():
    size = 0
    train_years = ["2010"]
    test_years = ["2017", "2018"]
    return [None]


def _get_data_func_adult_drift2(years):
    return X_train0, X_test20140, X_test20150, X_train, X_test2014 , X_test2015, y_train, y_test2014, y_test2015, strain, stest2014, stest2015, column_names"""





def _get_data_func_adult_drift(empty):

    test2014 = pd.read_csv("../data/adult_2014_formatted.csv", index_col=0)
    test2015 = pd.read_csv("../data/adult_2015_formatted.csv", index_col=0)
    train = pd.read_csv("../data/adult_original_formatted.csv", index_col=0)
    test = pd.read_csv("../data/adult_original_test_formatted.csv", index_col=0)

    y2014 = pd.read_csv("../data/adult_2014_y.csv", index_col=0)
    y2015 = pd.read_csv("../data/adult_2015_y.csv", index_col=0)
    ytrain = pd.read_csv("../data/adult_original_y.csv", index_col=0)
    ytest = pd.read_csv("../data/adult_original_test_y.csv", index_col=0)
    
    strain = (train["sex"] == "Male").astype('int')
    stest = (test["sex"] == "Male").astype('int')
    stest2014 = (test2014["sex"] == "Male").astype('int')
    stest2015 = (test2015["sex"] == "Male").astype('int')
    y_train = (ytrain == '>50K').astype(int).values.flatten()
    y_test = (ytest == '>50K.').astype(int).values.flatten()
    y_test2014 = y2014.astype(int).values.flatten()
    y_test2015 = y2015.astype(int).values.flatten()

    train["data"] = ["train"] * train.shape[0]
    test["data"] = ["test"] * test.shape[0]
    test2014["data"] = ["2014"] * test2014.shape[0]
    test2015["data"] = ["2015"] * test2015.shape[0]
    all_df = pd.concat((train, test, test2014, test2015))
    all_df = (all_df
         .drop(columns=['sex'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, columns = ['race', 'workclass', 'education', 
                    'marital_status', 'occupation', 'relationship', "country"], drop_first=True))
    
    X_train = all_df[all_df["data"] == "train"]
    X_test = all_df[all_df["data"] == "test"]
    X_test2014 = all_df[all_df["data"] == "2014"]
    X_test2015 = all_df[all_df["data"] == "2015"]
    
    del X_train["data"]
    del X_test["data"]
    del X_test2014["data"]
    del X_test2015["data"]
    del all_df["data"]

    scaler = MinMaxScaler().fit(all_df)
    
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train, X_test, X_test2014, X_test2015 = X_train.pipe(scale_df, scaler), X_test.pipe(scale_df, scaler), X_test2014.pipe(scale_df, scaler), X_test2015.pipe(scale_df, scaler)
    
    X_train, X_test, X_test2014, X_test2015 = X_train.values, X_test.values, X_test2014.values, X_test2015.values

    X_train0 = train[[c for c in train.columns if c not in ["sex", "data"]]]
    X_test0 = test[[c for c in test.columns if c not in ["sex", "data"]]]
    X_test20140 = test2014[[c for c in test2014.columns if c not in ["sex", "data"]]]
    X_test20150 = test2015[[c for c in test2015.columns if c not in ["sex", "data"]]]
    
    column_names = [c for c in test2015.columns if c not in ["sex", "data"]]
    return X_train0, X_test0, X_test20140, X_test20150, X_train, X_test, X_test2014 , X_test2015, y_train, y_test, y_test2014, y_test2015, strain, stest, stest2014, stest2015, column_names




def _get_data_func_german(empty):
    column_names = [
        'status', 'months', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
        'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
        'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone',
        'foreign_worker', 'credit'
    ]
    data_file="../data/german.data"
    dataset = pd.read_csv(data_file, sep=' ', header=None, names=column_names)
    personal_status_map = {'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'}
    dataset['sex'] = dataset['personal_status'].replace(personal_status_map)
    dataset.drop('personal_status', axis=1, inplace=True)
    features, labels = dataset.drop('credit', axis=1), dataset['credit']

    protected_att = 'sex'
    protected_unique = features[protected_att].nunique()
    protected = np.logical_not(pd.Categorical(features[protected_att]).codes)
    features=features.drop("sex", axis=1)
        
    continuous_vars = []
    categorical_columns = []
    for col in features.columns:
        if features[col].isnull().sum() > 0:
            features.drop(col, axis=1, inplace=True)
        else:
            if features[col].dtype == object:
                categorical_columns += [col]
            else:
                continuous_vars += [col]


    #print(features)
    features2=features
    #print(features.columns())
    print(features2)
    #del features["sex"]
    features = pd.get_dummies(features, columns=categorical_columns, prefix_sep='=')
    continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

    one_hot_columns = {}
    for column_name in categorical_columns:
        ids = [i for i, col in enumerate(features.columns) if col.startswith('{}='.format(column_name))]
        if len(ids) > 0:
            assert len(ids) == ids[-1] - ids[0] + 1
        one_hot_columns[column_name] = ids
    print('categorical features: ', one_hot_columns.keys())

    column_ids = {col: idx for idx, col in enumerate(features.columns)}

    features = features.values.astype(np.float32)
    labels = 2 - labels.values.astype(np.int64)
    protected = protected


    
    X_train0, X_test0, X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(features2,features, labels, protected, test_size=0.3)

    X_train=pd.DataFrame(data=X_train)
    X_test=pd.DataFrame(data=X_test)
    print(X_train.shape)
    scaler = MinMaxScaler().fit(X_train)
    scale_df = lambda df0, scaler: pd.DataFrame(scaler.transform(df0), columns=df0.columns, index=df0.index)
    X_train, X_test = np.array(X_train.pipe(scale_df, scaler)), np.array(X_test.pipe(scale_df, scaler))
    

    
    #scaler = MinMaxScaler().fit(X_train0)
    #scale_df = lambda df0, scaler: pd.DataFrame(scaler.transform(df0), columns=df0.columns, index=df0.index)
    #X_train, X_test = X_train0.pipe(scale_df, scaler), X_test0.pipe(scale_df, scaler)

    #X_train, X_test = X_train.values, X_test.values

    #column_names = list(X_train0.columns)

    return X_train0, X_test0, X_train, X_test, y_train, y_test, S_train, S_test, column_names





def _get_data_func_lawschool(empty):
    FNAME = '../data/lawschool.csv'
    df = pd.read_csv(FNAME)
    
    y = df["bar1"]
    del df["bar1"]
    S = df["race7"]
    del df["race1"],df["race2"],df["race3"],df["race4"],df["race5"],df["race6"],df["race7"],df["race8"]
    #,"race2","race3","race4","race5","race6","race7","race8"]]
    
    df["age"] = -1 * df["age"]
        
    X_train0, X_test0, y_train, y_test, S_train, S_test = train_test_split(df, y, S, test_size=0.3)
   
    scaler = MinMaxScaler().fit(X_train0)
    scale_df = lambda df0, scaler: pd.DataFrame(scaler.transform(df0), columns=df0.columns, index=df0.index)
    X_train, X_test = X_train0.pipe(scale_df, scaler), X_test0.pipe(scale_df, scaler)

    X_train, X_test = X_train.values, X_test.values

    column_names = list(X_train0.columns)
    
    return X_train0, X_test0, X_train, X_test, y_train, y_test, S_train, S_test, column_names
    
    
def _get_data_func_adult(empty):
    # load ICU data set
    XD_train0, XC_train0, y_train, S_train = load_ICU_data('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    XD_test0, XC_test0, y_test, S_test = load_ICU_data('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')
        
    #make up for inconsistency in columns between train and test (one col diff)
    val=XC_test0['age']*0
    ind= XC_train0.columns.get_loc(XC_train0.columns.difference(XC_test0.columns).values[0])
    name= XC_train0.columns.difference(XC_test0.columns).values[0]
    XC_test0.insert(loc=ind, column= name, value=val)# there is a one column diff between train and test --> recreate it with zeros
    
    # scale data
    scaler = MinMaxScaler().fit(XD_train0)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    scalerNC = MinMaxScaler().fit(XC_train0) # why two scalers?
    scale_dfNC = lambda df, scalerNC: pd.DataFrame(scalerNC.transform(df), columns=df.columns, index=df.index)
    XD_train, XD_test = XD_train0.pipe(scale_df, scaler), XD_test0.pipe(scale_df, scaler)
    XC_train, XC_test = XC_train0.pipe(scale_dfNC, scalerNC), XC_test0.pipe(scale_dfNC, scalerNC)

    XD_train, XD_test, XC_train, XC_test, y_train, y_test = XD_train.values, XD_test.values, XC_train.values, XC_test.values, y_train.values, y_test.values
    S_train, S_test =S_train.values.squeeze(1), S_test.values.squeeze(1)
    
    X_train0 =  pd.concat((XC_train0,XD_train0), axis=1)
    X_test0 =  pd.concat((XC_test0,XD_test0), axis=1)
    X_train =  np.concatenate((XC_train,XD_train), axis=1)
    X_test =  np.concatenate((XC_test,XD_test), axis=1)
    column_names = list(XC_train0.columns) + list(XD_train0.columns)
    
    
    return X_train0, X_test0, X_train, X_test, y_train, y_test, S_train, S_test, column_names


def _get_data_func_compass(num,sens,y,columns_delete):
    dataset = DATASETS[num] # Adult data set?????
    all_sensitive_attributes = dataset.get_sensitive_attributes_with_joint()
    ProcessedData(dataset)
    processed_dataset = ProcessedData(dataset)
    train_test_splits = processed_dataset.create_train_test_splits(1)
    train_test_splits.keys()
    train, test = train_test_splits['numerical-binsensitive'][0]
    X_train = train
    X_test = test

    sensitive =  train[sens].values
    sensitivet =  test[sens].values
    y_train = train[y]
    y_test = test[y]

    scaler = StandardScaler().fit(X_train)
    s=np.expand_dims(X_train[sens],axis=1)
    st=np.expand_dims(X_test[sens],axis=1)
    t=X_train[y]
    tt=X_test[y]
    #XC_train0=X_train[['age','sex-race','race']]
    #XC_test0=X_test[['age','sex-race','race']]
    
    X_train0 = X_train.drop(columns_delete,1)
    X_test0 = X_test.drop(columns_delete,1)
    
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)
    X_train= X_train.drop([sens,y], axis=1)
    X_train[sens] = s # no sensitive in X?
    X_train[y] = t
    X_test= X_test.drop([sens,y], axis=1)
    X_test[sens] = st
    X_test[y] = tt
    
    #XC_train=X_train[['age','sex-race','race']].values
    #XC_test=X_test[['age','sex-race','race']].values
    
    X_train = X_train.drop(columns_delete,1).values
    X_test = X_test.drop(columns_delete,1).values
    ### X_train = tout sauf Y et S (et sex-race)
    
    column_names = X_train0.columns
    
    #removed XC 
    return X_train0, X_test0, X_train, X_test, y_train, y_test, sensitive, sensitivet, column_names



def load_ICU_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                    'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names, 
                              na_values="?", sep=r'\s*,\s*', engine='python', header=1))
                  #.loc[lambda df: df['race'].isin(['White', 'Black'])])
    #input_data = pd.concat([input_data, pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', names=column_names, na_values="?", sep=r'\s*,\s*', engine='python').loc[1:,:] ])

    input_data[['age','fnlwgt','education_num','capital_gain', 'capital_loss', 'hours_per_week']] = input_data[['age','fnlwgt','education_num','capital_gain', 'capital_loss', 'hours_per_week']].astype(int)
    #input_data = input_data.drop(columns=['hours_per_week'])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['sex']
    S = (input_data.loc[:, sensitive_attribs]
         .assign(sex=lambda df: (df['sex'] == 'Male').astype(int)))
    # targets; 1 when someone makes over 50k , otherwise 0
    y = input_data['target'].replace({'<=50K.': 0, '>50K.': 1, '>50K': 1, '<=50K': 0 })
    XC = input_data.loc[:, ['country','race','age']]
    #XC = input_data.loc[:, ['age','country','race']]
    XC = XC.assign(race=lambda df: (df['race'] == 'White').astype(int))
    #print(XC.shape)
    XC = (XC
           .fillna('Unknown')
           .pipe(pd.get_dummies, columns = ['country'], drop_first=True))
    # features; note that the 'target' and sentive attribute columns are dropped
    XD = (input_data
         #.drop(columns=['target','age','sex','country'])
         .drop(columns=['target','country','race','age','sex'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, columns = ['workclass', 'education', 
                    'marital_status', 'occupation', 'relationship'], drop_first=True))
    #X = X.drop(columns=['hours_per_week'])
    print(f"features XD: {XD.shape[0]} samples, {XD.shape[1]} attributes")
    print(f"features XC: {XC.shape[0]} samples, {XC.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    print(f"sensitives S: {S.shape[0]} samples, {S.shape[1]} attributes")
    return XD, XC, y, S


DATA_DICT_P = {'adult': _get_data_params_adult,
               'adult_drift': _get_data_params_adult_drift,
               #'adult_drift2': _get_data_params_adult_drift,
             'compass': _get_data_params_compass,
              'lawschool': _get_data_params_lawschool,
              'german': _get_data_params_german}

DATA_DICT_F = {'adult': _get_data_func_adult,
               'adult_drift': _get_data_func_adult_drift,
               #'adult_drift2': _get_data_func_adult_drift2,
               'compass': _get_data_func_compass,
              'lawschool': _get_data_func_lawschool,
              'german': _get_data_func_german}

