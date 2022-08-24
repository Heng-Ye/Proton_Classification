# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn import svm
#from xgboost import XGBClassifier # Or XGBRegressor for Logistic Regression
from xgboost import XGBRegressor # Or XGBClassifier for Classification
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mplhep as hep

import csv
import copy

import time
from datetime import timedelta

#import xgboost
#print(xgboost.__version__)

# Graphics in retina format are more sharp and legible
#%config InlineBackend.figure_format = 'retina'

# install packages if needed --------------
#pip install pandas if lack of package
#pip install xgboost if lack of package
#pip install matplotlib
#pip install seaborn
#pip install sklearn
#pip install xgboost==1.0.1
#pip install graphviz

# using Pandas.DataFrame data-format, other available format are XGBoost's DMatrix and numpy.ndarray

#HEP- Plot Style ------------------------------
#hep.style.use("CMS") # string aliases work too
#hep.style.use(hep.style.ROOT)
#hep.style.use(hep.style.ATLAS)

#start_time --------------------
start_time = time.monotonic()

#Read training and validation files ------------------
input_file_name='./data_prep/protons_mva2'
output_file_name='./models/lgbm.model'

X_train=pd.read_csv(input_file_name+'_X_train.csv')
X_valid=pd.read_csv(input_file_name+'_X_valid.csv')
y_train=pd.read_csv(input_file_name+'_y_train.csv')
y_valid=pd.read_csv(input_file_name+'_y_valid.csv')
z_valid=pd.read_csv(input_file_name+'_z_valid.csv')

#Read feature observables --------------------------------------------------------
feature_names=[c for c in X_train.columns if c not in ['train','tag','target']]

#Build LightGBM Regression Model ----------------------------------------
#hyper parameters
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    #'metric': ['l1','l2'],
    'metric': 'auc',	
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 100000
}


lgbm = lgb.LGBMRegressor(**hyper_params)
lgbm.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        early_stopping_rounds=1000)



#Save/output model ---------------------------------
model_lgbm.save_model(output_file_name)

#Calculate time spent in the training process --------------------------------------
end_time = time.monotonic()
print('Time spent of the code:', timedelta(seconds=end_time - start_time), " sec")
