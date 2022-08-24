# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
import xgboost as xgb
import lightgbm as lgbm
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
lgbm_train_data = lgbm.Dataset(X_train, label=y_train)
lgbm_valid_data = lgbm.Dataset(X_valid, label=y_valid)

parameters = {'objective': 'binary',
              'metric': 'auc',
              'is_unbalance': 'true',
              'boosting': 'gbdt',
              'num_leaves': 63,
              'feature_fraction': 0.5,
              'bagging_fraction': 0.5,
              'bagging_freq': 20,
              'learning_rate': 0.01,
              'verbose': 0
             }

model_lgbm = lgbm.train(parameters,
                            lgbm_train_data,
                            valid_sets=lgbm_valid_data,
                            num_boost_round=5000,
                            early_stopping_rounds=50)

#Save/output model ---------------------------------
model_lgbm.save_model(output_file_name)

#Calculate time spent in the training process --------------------------------------
end_time = time.monotonic()
print('Time spent of the code:', timedelta(seconds=end_time - start_time), " sec")
