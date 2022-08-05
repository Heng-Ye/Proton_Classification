# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
from xgboost import XGBClassifier # Or XGBRegressor for Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import csv
import copy

# Graphics in retina format are more sharp and legible
#%config InlineBackend.figure_format = 'retina'

# install packages if needed --------------
#pip install pandas if lack of package
#pip install xgboost if lack of package
#pip install matplotlib
#pip install seaborn
#pip install sklearn

# specify parameters via map
param = {'n_estimators':50}
xgb = XGBClassifier(param)

# using Pandas.DataFrame data-format, other available format are XGBoost's DMatrix and numpy.ndarray

#read train and test csv --------------------------------------------------------------------------------
#train_data = pd.read_csv("protons_mva2_train.csv", sep='\t') #use \t to separate
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',') 
#train_data = pd.read_csv("protons_mva2_train.csv", header=None) # no header
#train_data = pd.read_csv("protons_mva2_train.csv",usecols=[1,2]) # Only reads col1, col2.
#train_data = pd.read_csv("protons_mva2_train.csv",usecols=['ntrklen']) # Only reads 'ntrklen' 
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',', header=None, skiprows=1) #skip the top row 
train_data_raw = pd.read_csv("protons_mva2_train.csv", sep=',', header=None)
test_data_raw = pd.read_csv("protons_mva2_test.csv", sep=',', header=None)

train_data = train_data_raw.T #transpose
test_data = test_data_raw.T

print(train_data_raw.shape)
print(train_data.shape)


#header_train_data=train_data.columns #get header

#train_Variable = train_data['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''
train_Score = train_data['Type'] # Score should be integer, 0, 1, (2 and larger for multiclass)

test_Variable = test_data['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
test_Score = test_data['Type']

# Now the data are well prepared and named as train_Variable, train_Score and test_Variable, test_Score.
xgb.fit(train_Variable, train_Score) # Training
xgb.predict(test_Variable) # Outputs are integers
xgb.predict_proba(test_Variable) # Output scores , output structre: [prob for 0, prob for 1,...]
xgb.save_model("xgb.model") # Saving model


'''




#simple check -------------------------
#print(train_data)
#print(train_data.columns)
#print(train_data.head(3))
#print(header_train_data) 
#print(train_data.shape)
#train_data.tail(6) #last 6 elements

#train_arr=train_data.to_numpy()

#data cleaning

#plot results
#sns.countplot(x='trklen', data=train_data)
#plt.show()
#print(train_data.set_index('trklen'))
#train_data.set_index(['trklen'],['PID']).plot()
#plt.show()

#fit a curve

#save plots


