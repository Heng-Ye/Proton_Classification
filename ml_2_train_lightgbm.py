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
from argparse import ArgumentParser as ap

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
#input_file_name='./data_prep/protons_mva2'
#output_file_name='./models/lgbm.model'
#input_file_name='./data_prep/protons_mva2_20220902'
#output_file_name='./models/lgbm_mva2_20220902.model'

#Read file, feature observables, output file ------------------------------------------------------
parser = ap()
parser.add_argument("-d", type=str, help='Data File', default = "")
parser.add_argument("-f", type=str, help='Feature variables:', default = "'train','tag','target'")
parser.add_argument("-o", type=str, help='Output model name:', default = "")

args=parser.parse_args()
if not (args.d and args.f and args.o):
  print("--> Please provide input data file, feature observables, and output model name")
  exit()

input_file_name=args.d
feature_obs='['+args.f+']'
feature_obs_del = (args.f).split(',')
output_file_name=args.o

print('feature_obs:',feature_obs)

X_train=pd.read_csv(input_file_name+'_X_train.csv')
X_valid=pd.read_csv(input_file_name+'_X_valid.csv')
y_train=pd.read_csv(input_file_name+'_y_train.csv')
y_valid=pd.read_csv(input_file_name+'_y_valid.csv')
z_valid=pd.read_csv(input_file_name+'_z_valid.csv')

#Remove the unwanted features ------------------------------------
for col_each in feature_obs_del:
  col_find = [col for col in X_train.columns if col_each in col]
  #print('col_find:',col_find)
  if bool(col_find):
    del X_train[col_each[1:-1]]
    del X_valid[col_each[1:-1]]
    #print('col_each:',col_each)
  else:
    print('accepting all input features!')


#Read feature observables --------------------------------------------------------
#feature_names=[c for c in X_train.columns if c not in ['train','tag','target']]
feature_names=[c for c in X_train.columns if c not in [args.f]]

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


model_lgbm = lgb.LGBMRegressor(**hyper_params)
model_lgbm.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        early_stopping_rounds=1000)



#Save/output model ---------------------------------
model_lgbm.booster_.save_model(output_file_name)

#Calculate time spent in the training process --------------------------------------
end_time = time.monotonic()
print('Time spent of the code:', timedelta(seconds=end_time - start_time), " sec")
