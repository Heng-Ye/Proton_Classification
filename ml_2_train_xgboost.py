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
output_file_name='./models/xgb.model'

X_train=pd.read_csv(input_file_name+'_X_train.csv')
X_valid=pd.read_csv(input_file_name+'_X_valid.csv')
y_train=pd.read_csv(input_file_name+'_y_train.csv')
y_valid=pd.read_csv(input_file_name+'_y_valid.csv')
z_valid=pd.read_csv(input_file_name+'_z_valid.csv')

#Read feature observables --------------------------------------------------------
feature_names=[c for c in X_train.columns if c not in ['train','tag','target']]

#Build XGBoost Regression Model ----------------------------------------
#print out all parameters that can be used/tunned
print(xgb.XGBRegressor().get_params())

model_xgb = xgb.XGBRegressor(learning_rate=0.1,
                                      max_depth=5,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
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

#evaluation data set ------------
eval_set = [(X_valid, y_valid)]

#Call the fit func. to train the model -------------------------------------------------
model_xgb.fit(X_train,y_train,early_stopping_rounds=10,eval_set=eval_set,verbose=True)

#Save/output model ---------------------------------
model_xgb.save_model(output_file_name)

#Calculate time spent in the training process --------------------------------------
end_time = time.monotonic()
print('Time spent of the code:', timedelta(seconds=end_time - start_time), " sec")
