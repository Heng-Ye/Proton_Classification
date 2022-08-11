# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
import xgboost
from xgboost import XGBClassifier # Or XGBRegressor for Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import csv
import copy

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

# using Pandas.DataFrame data-format, other available format are XGBoost's DMatrix and numpy.ndarray

#read train and test csv --------------------------------------------------------------------------------
#train_data = pd.read_csv("protons_mva2_train.csv", sep='\t') #use \t to separate
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',') 
#train_data = pd.read_csv("protons_mva2_train.csv", header=None) # no header
#train_data = pd.read_csv("protons_mva2_train.csv",usecols=[1,2]) # Only reads col1, col2.
#train_data = pd.read_csv("protons_mva2_train.csv",usecols=['ntrklen']) # Only reads 'ntrklen' 
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',', header=None, skiprows=1) #skip the top row 
#train_data = pd.read_csv("protons_mva2_train.csv", sep=',', header=None)
train_data = pd.read_csv('protons_mva2.csv') #read all data
#test_data = pd.read_csv('protons_mva2_test.csv')
#X, y = train_data.iloc[:,:-1],train_data.iloc[:,-1]

#print header info ----------------
print(train_data.head(3))
#print(train_data.keys())
#print(train_data.shape)
#print(train_data.feature_names)
#train_data.info()
#train_data.describe()

#rename header ---------------------
#train_data2=train_data.rename(columns={'tag':'tag', 'ntrklen':'0', 'trklen':'1', 'PID':'2', 'B':'3', 'costheta':'4', 'mediandedx':'5', 'endpointdedx':'6', 'calo':'7', 'avcalo':'8'}, axis=1, inplace=True)
#print(train_data2.head(3))

#Split the columns into 'target(1:signal 0:background)' and the varable columns for training 
var_colums=[c for c in train_data.columns if c not in ['tag','target']]
X=train_data.loc[:, var_colums]
y=train_data.loc[:, 'target']

print(X.head(3))
print(y.head(3))

#split the whole data set into train and test set using 'train_test_split' function in sklearn ----
#test_size=percentage of events to be used for validation
split_frac=0.5
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_frac)
print('Dimensions:',X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

#Create XGBoost Classifier Model -----------------------------------------------------------------------------------------------------------------------------------------
#print out all parameters that can be used/tunned
print(xgboost.XGBClassifier().get_params())

model_xgboost = xgboost.XGBClassifier(learning_rate=0.1,
                                      max_depth=5,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      eval_metric='auc',
                                      verbosity=1)
#hyperparameters:
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

#Call the fit func. to train the model --------
model_xgboost.fit(X_train,
                  y_train,
                  early_stopping_rounds=10,
                  eval_set=eval_set,
                  verbose=True)

#early_stopping_rounds=10 means if the model performance does not change on the validation data set for 10 iterations then the training will stop
#which means no new tree will be built if the model performance does not change for 10 iterations

#Evaluate the performance -------------------------------------------------------------------------
#Get probability of preditions for every observation that that has been supplied to it
y_train_pred = model_xgboost.predict_proba(X_train)[:,1]
y_valid_pred = model_xgboost.predict_proba(X_valid)[:,1]
#predict prob provides 2 values for every observation that we provide to it: 
#1st value: prob. of that observation being a 0
#2nd value: prob. of that observation being a 1
#We are working with prediction prob of 1, so index is chosed to be 1 here

print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    roc_auc_score(y_valid, y_valid_pred)))

#Hyperparameter Tuning --------------------------



# specify parameters via map
#param = {'n_estimators':50}
#xgb = XGBClassifier(param)






#train_data = train_data_raw.T #transpose
#test_data = test_data_raw.T

#print(train_data_raw.shape)
#print(train_data.shape)


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


