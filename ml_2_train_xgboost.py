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
from argparse import ArgumentParser as ap
#from csv import reader
from io import StringIO

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
feature_obs=args.f
feature_obs_del = (args.f).split(',')
output_file_name=args.o

#feature_del_arr=[]
#feature_del_arr.append(args.f)
#feature_del_arr=pd.DataFrame( list(reader(args.f)))
#feature_del_arr = StringIO(args.f)
#df = pd.read_clipboard(feature_del_arr, sep =",")
#print(df)

#start_time --------------------
start_time = time.monotonic()

#Read training and validation files ------------------
#input_file_name='./data_prep/protons_mva2_20220902'
#input_file_name='./data_prep/protons_mva2'
#output_file_name='./models/xgb_mva2_20220902.model'

X_train=pd.read_csv(input_file_name+'_X_train.csv')
X_valid=pd.read_csv(input_file_name+'_X_valid.csv')
y_train=pd.read_csv(input_file_name+'_y_train.csv')
y_valid=pd.read_csv(input_file_name+'_y_valid.csv')
z_valid=pd.read_csv(input_file_name+'_z_valid.csv')

#Remove the unwanted features ----------
for col_each in feature_obs_del:
  del X_train[col_each[1:-1]]
  del X_valid[col_each[1:-1]]

#print(X_train.head(3))

#print('col_names[0]:',col_names[0])
#print('feature_del_arr:',feature_del_arr)
#print(X_train.keys())

#X_valid.drop(columns=feature_del_arr, axis='columns', inplace=True)
#X_train.drop(columns=[col_names], axis=1, inplace=True)
#print(X_valid.head(3))


#del X_train[feature_obs_del]
#del X_valid[feature_obs_del]

#Read feature observables --------------------------------------------------------
#feature_names=[c for c in X_train.columns if c not in ['train','tag','target']]
feature_names=[c for c in X_train.columns if c not in feature_obs]
print('feature_names used for training:',feature_names)

#Build XGBoost Regression Model ----------------------------------------
#print out all parameters that can be used/tunned
print(xgb.XGBRegressor().get_params())

model_xgb = xgb.XGBRegressor(learning_rate=0.1,
                                      max_depth=10,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
				      min_child_weight=1000,	
				      eta=1,	
                                      eval_metric='auc',
				      feature_names=feature_names,
                                      verbosity=1)

#Notes of hyperparameters:
#learning_rate: tells the model the weightage of every tree in XGBoost classifier
#max_depth: maximum depth for each tree
#n_estimators: the model will create maximum of N trees (N=5000 in this case)
#subsample: subsample=0.5 means that 50 percent of the observations in the training dataset will be randomly selected for creating each individual tree in the model 
#and every iteration will have different sample of observations. It helps us train the model quickly and also prevents overfitting
#subsample is a parameter for observations and we have a similiar parameter for the features that we have in our data, that is called 'colsample_bytree'.
#colsample_bytree: half of the features will be used every time randomly selected when ever a new tree is built in the model
#eval_metric='auc' which means that while the model is being trained it will be evaluated using area under the curve as a metric
#verbosity=1 means print out the log
#min_child_weight[default=1]: The larger min_child_weight is, the more conservative the algorithm will be.

#evaluation data set ------------
eval_set = [(X_valid, y_valid)]

#Call the fit func. to train the model -------------------------------------------------
model_xgb.fit(X_train,y_train,early_stopping_rounds=10,eval_set=eval_set,verbose=True)

#Save/output model ---------------------------------
model_xgb.save_model(output_file_name)

#Calculate time spent in the training process --------------------------------------
end_time = time.monotonic()
print('Time spent of the code:', timedelta(seconds=end_time - start_time), " sec")

print('feature_names used for training:',feature_names)
print('f_names=',model_xgb.get_booster().feature_names)
