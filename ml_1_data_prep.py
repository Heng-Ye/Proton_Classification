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

#start_time
#start_time = time.monotonic()

#read train and test csv --------------------------------------------------------------------------------
#train_data = pd.read_csv("protons_mva2_train.csv", sep='\t') #use \t to separate
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',') 
#train_data = pd.read_csv("protons_mva2_train.csv", header=None) # no header
#train_data = pd.read_csv("protons_mva2_train.csv",usecols=[1,2]) # Only reads col1, col2.
#train_data = pd.read_csv("protons_mva2_train.csv",usecols=['ntrklen']) # Only reads 'ntrklen' 
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',', header=None, skiprows=1) #skip the top row 
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',', header=None)

input_file_name='./data_prep/protons_mva2'
train_data = pd.read_csv(input_file_name+'.csv') #read input file
#train_data = pd.read_csv('protons_with_additionalinfo_mva2.csv') #read all data
#test_data = pd.read_csv('protons_mva2_test.csv')
#X, y = train_data.iloc[:,:-1],train_data.iloc[:,-1]

#print header info ----------------
print(train_data.head(3))
#print(train_data.keys())
#print(train_data.shape)
#print(train_data.feature_names)
#train_data.info()
#train_data.describe()

#rename header -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#train_data2=train_data.rename(columns={'tag':'tag', 'ntrklen':'0', 'trklen':'1', 'PID':'2', 'B':'3', 'costheta':'4', 'mediandedx':'5', 'endpointdedx':'6', 'calo':'7', 'avcalo':'8'}, axis=1, inplace=True)
#print(train_data2.head(3))

# Basic info of the data set ------------------------------
#Manually split the data
#frac_split=0.5
#n_s=len(train_data[train_data.tag==1])
#n_b=len(train_data[train_data.tag!=1])
#n_b_el=len(train_data[train_data.tag==2])
#n_b_midp=len(train_data[train_data.tag==5])
#n_all=(float)(n_s+n_b)

#frac_s=n_s/n_all
#frac_el=n_b_el/n_all
#frac_midp=n_b_midp/n_all
#n_split=(int)(n_all*frac_split)

#print('Number of signal (inel.) events: {}'.format(n_s))
#print('Number of background events: {}'.format(n_b))
#print('Number of all events: {}'.format(n_all))
#print('Fraction signal: {}'.format(frac_s))
#print('Fraction bkg (el): {}'.format(frac_el))
#print('Fraction bkg (misidp): {}'.format(frac_midp))
#print('N_split: {}'.format(n_split))

#Split data manually (some errors in the training processes, FIXME) -----------------------------------------
#X_train=X[:n_split]
#X_valid=X[n_split:]
#y_train=y[:n_split]
#y_valid=y[:n_split]
#train=xgb.DMatrix(data=X_train[feature_names],label=y_train,missing=-999.0,feature_names=feature_names)
#valid=xgb.DMatrix(data=X_valid[feature_names],label=y_valid,missing=-999.0,feature_names=feature_names)
#Bad F-score performance using manual spliting, not sure what's the actual reason


#Split the columns into 'target(1:signal 0:background)' and the varable columns for training -----------------------------------------------
var_colums=[c for c in train_data.columns if c not in ['train','tag','target']]
#var_colums=[c for c in train_data.columns if c not in ['train','tag','target','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd']]
X=train_data.loc[:, var_colums]
y=train_data.loc[:, 'target']
z=train_data.loc[:, 'tag']
feature_names = var_colums

print(X.head(3))
print(y.head(3))

#Split the whole data set into train and test set using 'train_test_split' function in sklearn -----------------------------------------
#test_size=percentage of events to be used for validation
split_frac=0.4
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_frac)
X_train, X_valid, z_train, z_valid = train_test_split(X, z, test_size=split_frac)
#[Note]'train_test_split' will convert the dataframe to numpy array which dont have columns information anymore(ie, no feature names).

#Replace badground tags to value 0 -----------
#We still want to preserve the original tags
y_train=z_train.replace([2,3,4,5,6,7,8], 0)           
y_valid=z_valid.replace([2,3,4,5,6,7,8], 0)

#print(z_valid.head(10))
#print(y_valid.head(10))

print('Dimensions:',X_train.shape, X_valid.shape, y_train.shape, y_valid.shape, z_valid.shape)
print('Number of training samples: {}'.format(len(X_train)))
print('Number of cross-validation samples: {}'.format(len(X_valid)))

#Save the traing and validation sets -----------------------
X_train.to_csv(input_file_name+'_X_train.csv', index=False)
X_valid.to_csv(input_file_name+'_X_valid.csv', index=False)
y_train.to_csv(input_file_name+'_y_train.csv', index=False)
y_valid.to_csv(input_file_name+'_y_valid.csv', index=False)
z_valid.to_csv(input_file_name+'_z_valid.csv', index=False)




