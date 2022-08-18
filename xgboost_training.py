# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
import xgboost as xgb
#from xgboost import XGBClassifier # Or XGBRegressor for Logistic Regression
from xgboost import XGBRegressor # Or XGBClassifier for Classification
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
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
hep.style.use("CMS") # string aliases work too
#hep.style.use(hep.style.ROOT)
#hep.style.use(hep.style.ATLAS)

#start_time
start_time = time.monotonic()

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

#rename header -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#train_data2=train_data.rename(columns={'tag':'tag', 'ntrklen':'0', 'trklen':'1', 'PID':'2', 'B':'3', 'costheta':'4', 'mediandedx':'5', 'endpointdedx':'6', 'calo':'7', 'avcalo':'8'}, axis=1, inplace=True)
#print(train_data2.head(3))

# Basic info of the data set ------------------------------
frac_split=0.5
n_s=len(train_data[train_data.tag==1])
n_b=len(train_data[train_data.tag!=1])

n_b_el=len(train_data[train_data.tag==2])
n_b_midp=len(train_data[train_data.tag==5])
n_all=(float)(n_s+n_b)

frac_s=n_s/n_all
frac_el=n_b_el/n_all
frac_midp=n_b_midp/n_all
n_split=(int)(n_all*frac_split)

print('Number of signal (inel.) events: {}'.format(n_s))
print('Number of background events: {}'.format(n_b))
print('Number of all events: {}'.format(n_all))
print('Fraction signal: {}'.format(frac_s))

print('Fraction bkg (el): {}'.format(frac_el))
print('Fraction bkg (misidp): {}'.format(frac_midp))

print('N_split: {}'.format(n_split))

#Split the columns into 'target(1:signal 0:background)' and the varable columns for training --
var_colums=[c for c in train_data.columns if c not in ['train','tag','target']]
X=train_data.loc[:, var_colums]
y=train_data.loc[:, 'target']
z=train_data.loc[:, 'tag']
feature_names = var_colums

print(X.head(3))
print(y.head(3))

#split the whole data set into train and test set using 'train_test_split' function in sklearn -----------------------------------------
#test_size=percentage of events to be used for validation
split_frac=0.4
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_frac)
X_train, X_valid, z_train, z_valid = train_test_split(X, z, test_size=split_frac)
#[Note]'train_test_split' will convert the dataframe to numpy array which dont have columns information anymore(ie, no feature names).

#replace badground tags to value 0 -----------
#We still want to preserve the original tags
y_train=z_train.replace([2,3,4,5,6,7,8], 0)           
y_valid=z_valid.replace([2,3,4,5,6,7,8], 0)

#print(z_valid.head(10))
#print(y_valid.head(10))

print('Dimensions:',X_train.shape, X_valid.shape, y_train.shape, y_valid.shape, z_valid.shape)
print('Number of training samples: {}'.format(len(X_train)))
print('Number of cross-validation samples: {}'.format(len(X_valid)))

#Split the data manually -------------------------------------------------------------------------------
#X_train=X[:n_split]
#X_valid=X[n_split:]
#y_train=y[:n_split]
#y_valid=y[:n_split]
#train=xgb.DMatrix(data=X_train[feature_names],label=y_train,missing=-999.0,feature_names=feature_names)
#valid=xgb.DMatrix(data=X_valid[feature_names],label=y_valid,missing=-999.0,feature_names=feature_names)
#Bad F-score performance using manual spliting, not sure what's the actual reason

#Create XGBoost Classifier Model -----------------------------------------------------------------------------------------------------------------------------------------
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
model_xgb.save_model("xgb_prebuilt.model")

#Evaluate the performance ------------------------------------------------------------------------------
#Get probability of preditions for every observation that that has been supplied to it
#y_train_pred = model_xgb.predict_proba(X_train)[:,1]
#y_valid_pred = model_xgb.predict_proba(X_valid)[:,1]
#predict prob provides 2 values for every observation that we provide to it: 
#1st value: prob. of that observation being a 0
#2nd value: prob. of that observation being a 1
#We are working with prediction prob of 1, so index is chosed to be 1 here

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
#XGBRegressor' object has no attribute 'predict_proba (the following lines only work for classification)
#print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
#                                                    roc_auc_score(y_valid, y_valid_pred)))

#y_train_bkg_pred = model_xgboost.predict_proba(X_train)[:,0]
#y_valid_bkg_pred = model_xgboost.predict_proba(X_valid)[:,0]
#print("AUC Train(bkg): {:.4f}\nAUC Valid(bkg): {:.4f}".format(roc_auc_score(y_train, y_train_bkg_pred),
#                                                    roc_auc_score(y_valid, y_valid_bkg_pred)))

#Feature importance plot ---------------------------------------------------------------------
plt.rcParams["figure.figsize"] = (16, 9)
xgb.plot_importance(model_xgb.get_booster(),grid=False)
#plt.subplots_adjust(left=0.212, right=0.943)
plt.subplots_adjust(left=0.150, right=0.943)
#plt.show()
plt.savefig("xgb_0_importance_plot_prebuilt_model.eps")
#plot_importance is based on matplotlib, so the plot can be saved use plt.savefig()
#print('feature_names',feature_names)

#Model training process, evaluation using auc  ---------------------------------------------
print(eval_set)
evaluation_results_model_xgb = model_xgb.evals_result()
# Index into each key to find AUC values for training and validation data after each tree
train_auc_tree_model_xgb  = evaluation_results_model_xgb['validation_0']['auc']
#valid_auc_tree_model_xgb  = evaluation_results_model_xgb['validation_1']['auc']

# Plotting Section
plt.figure(figsize=(12,9))
plt.plot(train_auc_tree_model_xgb, label='Training set')
#plt.plot(valid_auc_tree_model_xgb , label='valid')

#plt.title("Train and validation AUC as number of trees increase")
plt.title("Training Iteration")
plt.xlabel("Number of Trees")
plt.ylabel("AUC (Area under the ROC Curve)")
plt.legend(frameon=False,loc='center right')
#plt.show()
plt.savefig('xgb_1_AUC_train_tree_num_prebuilt.eps')

#Predict results ---------------------------------------------------------------------------------------------------
results=model_xgb.evals_result()
#print(results)

#Get model predictions for validation set --------
#predictions = model_xgb.predict(X_valid)
y_pred = model_xgb.predict(X_valid)

# evaluate model accuracy -----------------------------
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_valid, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#print(predictions)
print('type of X_valid:',type(X_valid))
print('type of y_pred:',type(y_pred))
print('type of predictions:',type(predictions))

# Add newly-produced BDT score to validation set ----------
print('Dimensions of X_valid:',X_valid.shape)
print(X_valid.head(3))
#bdt_score=['bdt_score', y_pred]
#print('Dimensions of bdt_score:', np.shape(bdt_score))

#Merge all info for valid set
XY_valid=X_valid.copy()
XY_valid.insert(0, "bdt_score", y_pred, True)
print(XY_valid.head(3))
print('Dimensions of y_pred:',y_pred.shape)
print('Dimensions of XY_valid:',XY_valid.shape)

XY_valid.insert(1, "target", y_valid, True)
XY_valid.insert(2, "tag", z_valid, True)
print(y_valid.head(3))
print(XY_valid.head(3))
print('Dimensions of y_valid:',y_valid.shape)
print('Dimensions of XY_valid:',XY_valid.shape)

# plot single tree ------------------------------
plot_tree(model_xgb, num_trees=0, rankdir='LR')
#plt.show()
plt.savefig('xgb_4_Decision_trees.eps')

# plot BDT score ----- ---------------------------------------------------------------------------------------------- 
plt.figure()
n_bdt=100
bdt_min=0
bdt_max=1
plt.figure(figsize = (12,9))
plt.title("XGBoost: Training Result") 
plt.hist(y_pred,bins=np.linspace(bdt_min,bdt_max,n_bdt),histtype='step',color='black',label='All events', log=True)

# make the plot readable
#plt.xlabel('BDT Score',fontsize=12)
#plt.ylabel('Events',fontsize=12)
plt.xlabel('BDT Score', loc='center')
plt.ylabel('Events', loc='center')
plt.legend(frameon=False)
#plt.savefig('BDT_predict.png')

# plot signal and background separately
#plt.figure()
#select true signal and background with 'target' label 1 and 0
bdt_strue=XY_valid.loc[XY_valid['target']==1]
bdt_btrue=XY_valid.loc[XY_valid['target']==0]

#select only the bdt_score column
bdt_s=bdt_strue.loc[:, 'bdt_score']
bdt_b=bdt_btrue.loc[:, 'bdt_score']

#print('Dimensions of bdt_strue:',bdt_strue.shape)
plt.hist(bdt_s,bins=np.linspace(bdt_min,bdt_max,n_bdt),histtype='step',color='blue',label='signal', log=True)
plt.hist(bdt_b,bins=np.linspace(bdt_min,bdt_max,n_bdt),histtype='step',color='red',label='background', log=True)

# make the plot readable
#plt.xlabel('Prediction from BDT',fontsize=12)
#plt.ylabel('Events',fontsize=12)
plt.legend(loc='upper center', frameon=False)
plt.savefig('xgb_2_BDT_predict.eps')


#Use BDT score to select inelastic-scattering events ---------------------
#evt selection cuts
bdt_cut=0.5

#select inel. events based on the BDT_score cut (in DataFrame)
XY_valid_inel=XY_valid.loc[XY_valid['bdt_score']>bdt_cut]
print(XY_valid_inel.head(3))

#Old evt selection cut using PID
XY_valid_inel_old=XY_valid.loc[XY_valid['PID']>10]


#statistics of old/new cuts -------------------------------------------------------------------------------
#old PID cut
nvalid_tot=len(XY_valid_inel_old)
nvalid_inel=len(XY_valid_inel_old.loc[XY_valid_inel_old['tag']==1])
nvalid_el=len(XY_valid_inel_old.loc[XY_valid_inel_old['tag']==2])
nvalid_misidp=len(XY_valid_inel_old.loc[XY_valid_inel_old['tag']==5])
nvalid_other=nvalid_tot-nvalid_inel-nvalid_el-nvalid_misidp

frac_inel=(float)(nvalid_inel/nvalid_tot)
frac_el=(float)(nvalid_el/nvalid_tot)
frac_misidp=(float)(nvalid_misidp/nvalid_tot)
frac_other=(float)(nvalid_other/nvalid_tot)

#New BDT cut
nnvalid_tot=len(XY_valid_inel)
nnvalid_inel=len(XY_valid_inel.loc[XY_valid_inel['tag']==1])
nnvalid_el=len(XY_valid_inel.loc[XY_valid_inel['tag']==2])
nnvalid_misidp=len(XY_valid_inel.loc[XY_valid_inel['tag']==5])
nnvalid_other=nnvalid_tot-nnvalid_inel-nnvalid_el-nnvalid_misidp

ffrac_inel=(float)(nnvalid_inel/nnvalid_tot)
ffrac_el=(float)(nnvalid_el/nnvalid_tot)
ffrac_misidp=(float)(nnvalid_misidp/nnvalid_tot)
ffrac_other=(float)(nnvalid_other/nnvalid_tot)


'''
nvalid_inel_bdt=len(XY_valid.loc[(XY_valid['bdt_score']>bdt_cut)&(XY_valid['target']==1)])
nvalid_inel_old=len(XY_valid.loc[(XY_valid['PID']>10)&(XY_valid['target']==1)])

#statistics of truth info
nvalid_s_true=len(bdt_strue)
nvalid_b_true=len(bdt_btrue)
nvalid_all_true=(float)(nvalid_s_true+nvalid_b_true)

frac_bdt=nvalid_inel_bdt/nvalid_s_true
frac_old=nvalid_inel_old/nvalid_s_true

#n_all=(float)(n_s+n_b)

#frac_s=n_s/n_all
#frac_el=n_b_el/n_all
#frac_midp=n_b_midp/n_all
#n_split=(int)(n_all*frac_split)
'''
print('==Old PID Cut ===================================================================')
print('Number of all events: {}'.format(nvalid_tot))
print('Number of inel events: {} ({:0.2f}%)'.format(nvalid_inel, 100.*frac_inel))
print('Number of el events: {} ({:0.2f}%)'.format(nvalid_el, 100.*frac_el))
print('Number of misid:p events: {} ({:0.2f}%)'.format(nvalid_misidp, 100.*frac_misidp))
print('Number of other events: {} ({:0.2f}%)'.format(nvalid_other, 100.*frac_other))
print('==After Cut ===================================================================')
print('Number of all events: {}'.format(nnvalid_tot))
print('Number of inel events: {} ({:0.2f}%)'.format(nnvalid_inel, 100.*ffrac_inel))
print('Number of el events: {} ({:0.2f}%)'.format(nnvalid_el, 100.*ffrac_el))
print('Number of misid:p events: {} ({:0.2f}%)'.format(nnvalid_misidp, 100.*ffrac_misidp))
print('Number of other events: {} ({:0.2f}%)'.format(nnvalid_other, 100.*ffrac_other))



#print('Number of BDT (inel.) events: {}'.format(nvalid_inel_bdt))
#print('Number of true signal (inel.) events: {}'.format(nvalid_all_true))
#print('Number of background events: {}'.format(nvalid_b_true))
#print('Number of all events: {}'.format(n_all))
#print('Purity of new bdt cut: {}'.format(frac_bdt))
#print('Purity of old chi2 cut: {}'.format(frac_old))


#1D trklen distribution: before/after new/old cuts -------------------------------------------------------------------------------------------------------------
#before cut ------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure()
n_trklen=65
trklen_min=0
trklen_max=130
bins_each=np.linspace(trklen_min,trklen_max,n_trklen)

plt.figure(figsize = (12,9))
plt.title("Validation Set")
plt.hist(XY_valid.loc[:, 'trklen'], bins=np.linspace(trklen_min,trklen_max,n_trklen), histtype='step', stacked=False, color='black', label='All', log=False)

plt.hist([\
XY_valid[XY_valid.tag==1].loc[:, 'trklen'],\
XY_valid[XY_valid.tag==2].loc[:, 'trklen'], \
XY_valid[XY_valid.tag==5].loc[:, 'trklen'], \
XY_valid.query('(tag!=1) & (tag!=2) & (tag!=5)').loc[:,'trklen']\
],
 bins_each, \
 histtype='stepfilled', \
 stacked=True, \
 label=['Inel.','El.','MisID:P','Others'],\
 color=['red','blue','green','yellow'],
 log=False)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [0,4,3,2,1]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', frameon=True)
plt.ylim(0, 1000)
plt.xlabel("Track Length [cm]")
plt.ylabel("Events")
plt.savefig('xgb_5_trklen_beforecut.eps')

#after cut ------------------------------------------------------------------------------------------------------------------------------------------------------------
#PID cut ----------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.figure(figsize = (12,9))
plt.title("Validation Set with $\chi^{2}$PID Cut")
plt.hist(XY_valid.loc[:, 'trklen'], bins=np.linspace(trklen_min,trklen_max,n_trklen), histtype='step', stacked=False, color='black', label='All', log=False)

plt.hist([\
XY_valid_inel_old[XY_valid_inel_old.tag==1].loc[:, 'trklen'],\
XY_valid_inel_old[XY_valid_inel_old.tag==2].loc[:, 'trklen'], \
XY_valid_inel_old[XY_valid_inel_old.tag==5].loc[:, 'trklen'], \
XY_valid_inel_old.query('(tag!=1) & (tag!=2) & (tag!=5)').loc[:,'trklen']\
],
 bins_each, \
 histtype='stepfilled', \
 stacked=True, \
 label=['Inel. ({:0.1f}%)'.format(100.*frac_inel),'El. ({:0.1f}%)'.format(100.*frac_el),'MisID:P ({:0.1f}%)'.format(100.*frac_misidp),'Others ({:0.2f}%)'.format(100.*frac_other)],\
 color=['red','blue','green','yellow'],
 log=False)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [0,4,3,2,1]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', frameon=True)
plt.ylim(0, 1000)
plt.xlabel("Track Length [cm]")
plt.ylabel("Events")
plt.savefig('xgb_6_trklen_aftercut_PID.eps')

#BDT cut
plt.figure()
plt.figure(figsize = (12,9))
plt.title("Validation Set with BDT Cut")
plt.hist(XY_valid.loc[:, 'trklen'], bins=np.linspace(trklen_min,trklen_max,n_trklen), histtype='step', stacked=False, color='black', label='All', log=False)

plt.hist([\
XY_valid_inel[XY_valid_inel.tag==1].loc[:, 'trklen'],\
XY_valid_inel[XY_valid_inel.tag==2].loc[:, 'trklen'], \
XY_valid_inel[XY_valid_inel.tag==5].loc[:, 'trklen'], \
XY_valid_inel.query('(tag!=1) & (tag!=2) & (tag!=5)').loc[:,'trklen']\
],
 bins_each, \
 histtype='stepfilled', \
 stacked=True, \
 label=['Inel. ({:0.1f}%)'.format(100.*ffrac_inel),'El. ({:0.1f}%)'.format(100.*ffrac_el),'MisID:P ({:0.1f}%)'.format(100.*ffrac_misidp),'Others ({:0.2f}%)'.format(100.*ffrac_other)],\
 color=['red','blue','green','yellow'],
 log=False)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [0,4,3,2,1]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', frameon=True)
plt.ylim(0, 1000)
plt.xlabel("Track Length [cm]")
plt.ylabel("Events")
plt.savefig('xgb_7_trklen_aftercut_BDT.eps')




#2D scatter plot -----------------------------------------------------------------------------------
x_ntrklen=XY_valid_inel.loc[:, 'ntrklen']
y_pid=XY_valid_inel.loc[:, 'PID']

plt.figure(figsize = (12,9))
plt.title("Event Selection of Inelastic-scattering Protons") 
plt.plot(XY_valid.loc[:, 'ntrklen'], XY_valid.loc[:, 'PID'], 'ko', markersize=1, label='All events')
plt.plot(x_ntrklen, y_pid,'ro', markersize=1, label='After event selection')
plt.xlim(0, 1.3)
plt.ylim(-.5, 250)

plt.xlabel("Normalized Track Length [a.u.]")
plt.ylabel("$\chi^{2}$ PID [a.u.]")
plt.legend(frameon=True)
plt.savefig('xgb_3_ntrklen_pid_inelcut.eps')
 

plt.figure(figsize = (12,9))
plt.title("Basic Cuts") 
plt.plot(XY_valid.loc[:, 'ntrklen'], XY_valid.loc[:, 'PID'], 'ko', markersize=1, label='All events')
plt.xlim(0, 1.3)
plt.ylim(-.5, 250)

plt.xlabel("Normalized Track Length [a.u.]")
plt.ylabel("$\chi^{2}$ PID [a.u.]")
plt.legend(frameon=True)
plt.savefig('xgb_3_ntrklen_pid_nocut.eps')





end_time = time.monotonic()
print('Time spent of the code:', timedelta(seconds=end_time - start_time), " sec")


