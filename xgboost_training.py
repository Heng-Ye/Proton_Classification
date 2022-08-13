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

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    roc_auc_score(y_valid, y_valid_pred)))

#y_train_bkg_pred = model_xgboost.predict_proba(X_train)[:,0]
#y_valid_bkg_pred = model_xgboost.predict_proba(X_valid)[:,0]
#print("AUC Train(bkg): {:.4f}\nAUC Valid(bkg): {:.4f}".format(roc_auc_score(y_train, y_train_bkg_pred),
#                                                    roc_auc_score(y_valid, y_valid_bkg_pred)))


#Hyperparameter Tuning --------------------------------------------------------------------------------
#Use 3 hyper pars (27 configs.) as an example
learning_rate_list = [0.02, 0.05, 0.1]
max_depth_list = [2, 3, 5]
n_estimators_list = [1000, 2000, 3000]

#transform pars into dict.
params_dict = {"learning_rate": learning_rate_list,
               "max_depth": max_depth_list,
               "n_estimators": n_estimators_list}

num_combinations = 1
for v in params_dict.values(): num_combinations *= len(v) 

print(num_combinations)
params_dict

def my_roc_auc_score(model, X, y): return roc_auc_score(y, model.predict_proba(X)[:,1])

model_xgboost_hp = GridSearchCV(estimator=xgboost.XGBClassifier(subsample=0.5,
                                                                colsample_bytree=0.25,
                                                                eval_metric='auc',
                                                                use_label_encoder=False),
                                param_grid=params_dict,
                                cv=4,
                                scoring=my_roc_auc_score,
                                return_train_score=True,
                                verbose=4)

model_xgboost_hp.fit(X, y)

#cv:=cross-validation:num. of samples of data will splitted (into 2 parts in this case) 
#in this case 2 iterations will be applied on data: 1st part for training, 2nd part for validation
#in general cv=3 or 4
#evaluation of model using roc_auc
#return_train_score=True:=as a output of x-validation, training score will also be printed
#verbose=4:=how much of the logs will be printed

df_cv_results = pd.DataFrame(model_xgboost_hp.cv_results_)
df_cv_results = df_cv_results[['rank_test_score','mean_test_score','mean_train_score',
                               'param_learning_rate', 'param_max_depth', 'param_n_estimators']]
df_cv_results.sort_values(by='rank_test_score', inplace=True)
df_cv_results


#Performance changes by chaning hyper parameters: plots ----------------------------------------------------------
# First sort by number of estimators as that would be x-axis
df_cv_results.sort_values(by='param_n_estimators', inplace=True)

# Find values of AUC for learning rate of 0.05 and different values of depth
lr_d2 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==2),:]
lr_d3 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==3),:]
lr_d5 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==5),:]
lr_d7 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.05) & (df_cv_results['param_max_depth']==7),:]

# Let us plot now
fig, ax = plt.subplots(figsize=(10,5))
lr_d2.plot(x='param_n_estimators', y='mean_test_score', label='Depth=2', ax=ax)
lr_d3.plot(x='param_n_estimators', y='mean_test_score', label='Depth=3', ax=ax)
lr_d5.plot(x='param_n_estimators', y='mean_test_score', label='Depth=5', ax=ax)
lr_d7.plot(x='param_n_estimators', y='mean_test_score', label='Depth=7', ax=ax)
plt.ylabel('Mean Validation AUC')
plt.title('Performance wrt # of Trees and Depth')
plt.savefig('xgboost_1_hyperparameter_tunning_Performance_wrt_num_of_Trees_and_Depth.png')
#in this case, learning_rate is kept const.
#x-axis: param_n_estimators=# of trees
#y-axis: Validation AUC

#Fine-tunning the hyper parameters -----------------------------------------------------------------------------------
# First sort by learning rate as that would be x-axis
df_cv_results.sort_values(by='param_learning_rate', inplace=True)

# Find values of AUC for learning rate of 0.05 and different values of depth
lr_t3k_d2 = df_cv_results.loc[(df_cv_results['param_n_estimators']==3000) & (df_cv_results['param_max_depth']==2),:]

# Let us plot now
fig, ax = plt.subplots(figsize=(10,5))
lr_t3k_d2.plot(x='param_learning_rate', y='mean_test_score', label='Depth=2, Trees=3000', ax=ax)
plt.ylabel('Mean Validation AUC')
plt.title('Performance wrt learning rate')
plt.savefig('xgboost_2_Performance_wrt_learning_rate.png')

#Final Model ---------------------------------------------------------------------------------------------------
#After hyper parameter tunning, we can pick up the best parameters to build our model
model_xgboost_fin = xgboost.XGBClassifier(learning_rate=0.05,
                                          max_depth=2,
                                          n_estimators=5000,
                                          subsample=0.5,
                                          colsample_bytree=0.25,
                                          eval_metric='auc',
                                          verbosity=1,
                                          use_label_encoder=False)

# Passing both training and validation dataset as we want to plot AUC for both
eval_set = [(X_train, y_train),(X_valid, y_valid)]


model_xgboost_fin.fit(X_train,
                  y_train,
                  early_stopping_rounds=20,
                  eval_set=eval_set,
                  verbose=True)

y_train_pred = model_xgboost_fin.predict_proba(X_train)[:,1]
y_valid_pred = model_xgboost_fin.predict_proba(X_valid)[:,1]

print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    roc_auc_score(y_valid, y_valid_pred)))


evaluation_results = model_xgboost_fin.evals_result()

# Index into each key to find AUC values for training and validation data after each tree
train_auc_tree = evaluation_results['validation_0']['auc']
valid_auc_tree = evaluation_results['validation_1']['auc']

# Plotting Section
plt.figure(figsize=(15,5))

plt.plot(train_auc_tree, label='Train')
plt.plot(valid_auc_tree, label='valid')

plt.title("Train and validation AUC as number of trees increase")
plt.xlabel("Trees")
plt.ylabel("AUC")
plt.legend(loc='lower right')
#plt.show()
plt.savefig('xgboost_3_AUC_tree_num.png')

#Feature importance using variable importance --------------------------------------------
df_var_imp = pd.DataFrame({"Variable": var_colums,
                           "Importance": model_xgboost_fin.feature_importances_}) \
                        .sort_values(by='Importance', ascending=False)
#df_var_imp[:10]
df_var_imp[:10]

#Save model
model_xgboost_fin.save_model("xgb.model")



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


