# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
import xgboost as xgb
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
var_colums=[c for c in train_data.columns if c not in ['train','tag','target']]
X=train_data.loc[:, var_colums]
y=train_data.loc[:, 'target']

print(X.head(3))
print(y.head(3))

#split the whole data set into train and test set using 'train_test_split' function in sklearn ----
#test_size=percentage of events to be used for validation
split_frac=0.4
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_frac)
print('Dimensions:',X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

#[Note]'train_test_split' will convert the dataframe to numpy array which dont have columns information anymore(ie, no feature names).
feature_names = var_colums
#X_train = pd.DataFrame(data=Xtrain, columns=feature_names)
#X_valid = pd.DataFrame(data=X_valid, columns=feature_names)
#Xval = pd.DataFrame(data=Xval, columns=feature_names)
#dtrain = xgb.DMatrix(Xtrain, label=ytrain)


#[Note]'train_test_split' will convert the dataframe to numpy array which dont have columns information anymore(ie, no feature names).
#X_train = pd.DataFrame(data=X_train, columns=var_colums.name)
#X_valid = pd.DataFrame(data=X_valid, columns=var_colums.name)


#Create XGBoost Classifier Model -----------------------------------------------------------------------------------------------------------------------------------------
#print out all parameters that can be used/tunned
print(xgb.XGBClassifier().get_params())


#model_xgb = xgb.XGBClassifier(learning_rate=0.1,
model_xgb = xgb.XGBRegressor(learning_rate=0.1,
                                      max_depth=5,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      eval_metric='auc',
				      feature_names=feature_names,
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
model_xgb.fit(X_train,
                  y_train,
                  early_stopping_rounds=10,
                  eval_set=eval_set,
                  verbose=True)

#early_stopping_rounds=10 means if the model performance does not change on the validation data set for 10 iterations then the training will stop
#which means no new tree will be built if the model performance does not change for 10 iterations

#Evaluate the performance -------------------------------------------------------------------------
'''
#Get probability of preditions for every observation that that has been supplied to it
y_train_pred = model_xgb.predict_proba(X_train)[:,1]
y_valid_pred = model_xgb.predict_proba(X_valid)[:,1]
#predict prob provides 2 values for every observation that we provide to it: 
#1st value: prob. of that observation being a 0
#2nd value: prob. of that observation being a 1
#We are working with prediction prob of 1, so index is chosed to be 1 here

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    roc_auc_score(y_valid, y_valid_pred)))

#y_train_bkg_pred = model_xgboost.predict_proba(X_train)[:,0]
#y_valid_bkg_pred = model_xgboost.predict_proba(X_valid)[:,0]
#print("AUC Train(bkg): {:.4f}\nAUC Valid(bkg): {:.4f}".format(roc_auc_score(y_train, y_train_bkg_pred),
#                                                    roc_auc_score(y_valid, y_valid_bkg_pred)))
'''

#Feature importance -----------------------------------------------------------------
xgb.plot_importance(model_xgb.get_booster(),grid=False)
plt.subplots_adjust(left=0.212, right=0.943)
#plt.show()
plt.savefig("importance_plot_prebuilt_model_.pdf")
#plot_importance is based on matplotlib, so the plot can be saved use plt.savefig()
#print('feature_names',feature_names)


evaluation_results_model_xgb = model_xgb.evals_result()
# Index into each key to find AUC values for training and validation data after each tree
train_auc_tree_model_xgb  = evaluation_results_model_xgb['validation_0']['auc']
#valid_auc_tree_model_xgb  = evaluation_results_model_xgb['validation_1']['auc']

# Plotting Section
plt.figure(figsize=(15,5))
plt.plot(train_auc_tree_model_xgb , label='Train')
#plt.plot(valid_auc_tree_model_xgb , label='valid')

#plt.title("Train and validation AUC as number of trees increase")
plt.title("Train AUC as number of trees increase")
plt.xlabel("Trees")
plt.ylabel("AUC")
plt.legend(loc='lower right')
#plt.show()
plt.savefig('xgboost_AUC_train_tree_num_prebuilt.png')

#Save model
model_xgb.save_model("xgb_prebuilt.model")

# Basic info of the data set ------------------------------
n_s=len(train_data[train_data.tag==1])
n_b=len(train_data[train_data.tag!=1])
n_b_el=len(train_data[train_data.tag==2])
n_b_midp=len(train_data[train_data.tag==5])

n_all=(float)(n_s+n_b)

frac_s=n_s/n_all
frac_el=n_b_el/n_all
frac_midp=n_b_midp/n_all

print('Number of signal (inel.) events: {}'.format(n_s))
print('Number of background events: {}'.format(n_b))
print('Fraction signal: {}'.format(frac_s))

print('Fraction bkg (el): {}'.format(frac_el))
print('Fraction bkg (misidp): {}'.format(frac_midp))

#Predic results -----------------------------------------------------------------------------------------
#print(model_xgb.evals_result())
predictions = model_xgb.predict(X_valid)
print(predictions)


# plot all predictions (both signal and background)
plt.figure();
plt.hist(predictions,bins=np.linspace(0,1,50),histtype='step',color='darkgreen',label='All events', log=False);
# make the plot readable
plt.xlabel('Prediction from BDT',fontsize=12);
plt.ylabel('Events',fontsize=12);
plt.legend(frameon=False);
plt.savefig('BDT_predict.png')

'''
# plot signal and background separately
plt.figure();
plt.hist(predictions[X_valid.get_label().astype(bool)],bins=np.linspace(0,1,50),
         histtype='step',color='midnightblue',label='signal');
plt.hist(predictions[~(X_valid.get_label().astype(bool))],bins=np.linspace(0,1,50),
         histtype='step',color='firebrick',label='background');
# make the plot readable
plt.xlabel('Prediction from BDT',fontsize=12);
plt.ylabel('Events',fontsize=12);
plt.legend(frameon=False);
plt.savefig('BDT_predict.png')
'''

