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

#HEP- Plot Style ------------------------------
hep.style.use("CMS") # string aliases work too
#hep.style.use(hep.style.ROOT)
#hep.style.use(hep.style.ATLAS)

#Read training and validation files ------------------
#input_file_name='./data_prep/protons_mva2'
#output_file_name='./models/xgb.model'
#output_path='./plts_perform/'
#output_inel_csv='./csv/xgb_inel_valid.csv'
#input_file_name='./data_prep/protons_mva2_20220902'
#output_file_name='./models/xgb_mva2_20220902.model'
#output_path='./plts_perform_20220902/'
#output_inel_csv='./csv_20220902/xgb_inel_valid.csv'

#Read file, feature observables, output file ------------------------------------------------------
parser = ap()
parser.add_argument("-d", type=str, help='Data File', default = "")
parser.add_argument("-f", type=str, help='Feature variables:', default = "'train','tag','target'")
parser.add_argument("-o", type=str, help='Output model name:', default = "")
parser.add_argument("-p", type=str, help='Output plot path:', default = "")
parser.add_argument("-ocsv", type=str, help='Output csv file:', default = "")


args=parser.parse_args()
if not (args.d and args.f and args.o):
  print("--> Please provide input data file, feature observables, and output model name")
  exit()

input_file_name=args.d
feature_obs='['+args.f+']'
feature_obs_del = (args.f).split(',')

output_file_name=args.o
output_path=args.p
output_inel_csv=args.ocsv

X_train=pd.read_csv(input_file_name+'_X_train.csv')
X_valid=pd.read_csv(input_file_name+'_X_valid.csv')
y_train=pd.read_csv(input_file_name+'_y_train.csv')
y_valid=pd.read_csv(input_file_name+'_y_valid.csv')
z_valid=pd.read_csv(input_file_name+'_z_valid.csv')

#Remove the unwanted features ----------
for col_each in feature_obs_del:
  del X_train[col_each[1:-1]]
  del X_valid[col_each[1:-1]]

#Read feature observables --------------------------------------------------------
#feature_names=[c for c in X_train.columns if c not in ['train','tag','target']]
feature_names=[c for c in X_train.columns if c not in [args.f]]


#Load model ------------------------------
model_xgb=xgb.XGBRegressor()
model_xgb.load_model(output_file_name)
model_xgb.get_booster().feature_names = feature_names
print('feature_names_in_model=',model_xgb.get_booster().feature_names)
print('feature_names:',feature_names)


#Evaluate model performance ------------------------------------------------------------------------------
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
plt.savefig(output_path+"xgb_0_importance_plot_xgb_model.eps")
#plot_importance is based on matplotlib, so the plot can be saved use plt.savefig()
#print('feature_names',feature_names)

#Correlation matrix ---------------------------------------------------------------------------
corr=X_train.corr()
plt.figure(figsize=(10,6))
plt.rcParams.update({'font.size': 10})
#label.set_fontsize(22)
#plt.rc('font', size=10)
sns.heatmap(corr, annot=True)
plt.savefig(output_path+"xgb_0_heatmap_plot_train.eps")
hep.style.use("CMS") # set back to default style


#Model training process, evaluation using auc  ---------------------------------------------
#[1] AUC vs tree num.
results = model_xgb.evals_result()
#print(results)

# Index into each key to find AUC values for training and validation data after each tree
train_auc_tree_model_xgb  = results['validation_0']['auc']
#valid_auc_tree_model_xgb  = results['validation_1']['auc']

plt.figure(figsize=(26,12))
plt.plot(train_auc_tree_model_xgb, label='XGBoost Training set')
#plt.plot(valid_auc_tree_model_xgb , label='valid')

#plt.title("Train and validation AUC as number of trees increase")
plt.title("Training Iteration")
plt.xlabel("Number of Trees")
plt.ylabel("AUC (Area under the ROC Curve)")
plt.legend(frameon=False,loc='center right')
#plt.show()
plt.savefig(output_path+'xgb_1_AUC_train_tree_num_prebuilt.eps')


#Get model predictions for validation set --------
y_pred = model_xgb.predict(X_valid)
#-------------------------------------------------

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

#[2] Merge all info for valid set ------------------
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

#[2] plot single tree -------------------------------
plot_tree(model_xgb, num_trees=0, rankdir='LR')
#plt.show()
plt.savefig(output_path+'xgb_2_Decision_trees.eps')

#[3] ROC and AUC should be obtained on test set -----------------------------------
fpr, tpr, _ = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show() # display the figure when not using jupyter display
plt.savefig(output_path+"xgb_3_roc.eps") # resulting plot is shown below


#[4] plot BDT score ----- ---------------------------------------------------------------------------------------------- 
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
plt.savefig(output_path+'xgb_4_BDT_predict.eps')



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


#nvalid_inel_bdt=len(XY_valid.loc[(XY_valid['bdt_score']>bdt_cut)&(XY_valid['target']==1)])
#nvalid_inel_old=len(XY_valid.loc[(XY_valid['PID']>10)&(XY_valid['target']==1)])

##statistics of truth info
#nvalid_s_true=len(bdt_strue)
#nvalid_b_true=len(bdt_btrue)
#nvalid_all_true=(float)(nvalid_s_true+nvalid_b_true)

#frac_bdt=nvalid_inel_bdt/nvalid_s_true
#frac_old=nvalid_inel_old/nvalid_s_true

#n_all=(float)(n_s+n_b)

#frac_s=n_s/n_all
#frac_el=n_b_el/n_all
#frac_midp=n_b_midp/n_all
#n_split=(int)(n_all*frac_split)

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


#[5] 1D trklen distribution: before/after new/old cuts -------------------------------------------------------------------------------------------------------------
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
plt.savefig(output_path+'xgb_5_trklen_beforecut.eps')

#[6]PID cut ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
plt.savefig(output_path+'xgb_6_trklen_aftercut_PID.eps')

#[7]BDT cut ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
plt.savefig(output_path+'xgb_7_trklen_aftercut_BDT.eps')




#[8]2D scatter plot -----------------------------------------------------------------------------------
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
plt.savefig(output_path+'xgb_8_ntrklen_pid_nocut.eps')


#2D scatter plot with truth labels ----------------------------------------------------------------------------------------------------------------
#[9]ntrklen vs pid ----------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.figure(figsize = (12,9))
plt.title("Event Selection with BDT Cut") 
#plt.plot(x_ntrklen, y_pid,'ro', markersize=1, label='After event selection')

#plt.plot(XY_valid.loc[:, 'ntrklen'], XY_valid.loc[:, 'PID'], 'ko', markersize=1, label='All events')

plt.plot(XY_valid_inel[XY_valid_inel.tag==1].loc[:,'ntrklen'], XY_valid_inel[XY_valid_inel.tag==1].loc[:,'PID'], 'ro', markersize=1, label='Inel.')
plt.plot(XY_valid_inel[XY_valid_inel.tag==2].loc[:,'ntrklen'], XY_valid_inel[XY_valid_inel.tag==2].loc[:,'PID'], 'bo', color='blue', markersize=4, label='El.')
plt.plot(XY_valid_inel[XY_valid_inel.tag==5].loc[:,'ntrklen'], XY_valid_inel[XY_valid_inel.tag==5].loc[:,'PID'], 'go', color='green', markersize=4, label='MisID:P')

plt.xlabel("Normalized Track Length [a.u.]")
plt.ylabel("$\chi^{2}$ PID [a.u.]")

plt.xlim(0, 1.3)
plt.ylim(-.5, 250)
plt.legend(frameon=True,loc='upper right')
plt.savefig(output_path+'xgb_9_ntrklen_pid_BDTcut.eps')

#[10]trklen vs pid-----------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.figure(figsize = (12,9))
plt.title("Event Selection with BDT Cut") 
#plt.plot(x_ntrklen, y_pid,'ro', markersize=1, label='After event selection')

#plt.plot(XY_valid.loc[:, 'ntrklen'], XY_valid.loc[:, 'PID'], 'ko', markersize=1, label='All events')

plt.plot(XY_valid_inel[XY_valid_inel.tag==1].loc[:,'trklen'], XY_valid_inel[XY_valid_inel.tag==1].loc[:,'PID'], 'ro', markersize=1, label='Inel.')
plt.plot(XY_valid_inel[XY_valid_inel.tag==2].loc[:,'trklen'], XY_valid_inel[XY_valid_inel.tag==2].loc[:,'PID'], 'bo', color='blue', markersize=4, label='El.')
plt.plot(XY_valid_inel[XY_valid_inel.tag==5].loc[:,'trklen'], XY_valid_inel[XY_valid_inel.tag==5].loc[:,'PID'], 'go', color='green', markersize=4, label='MisID:P')

plt.xlabel("Track Length [a.u.]")
plt.ylabel("$\chi^{2}$ PID [a.u.]")

plt.xlim(0, 120)
plt.ylim(-.5, 250)
plt.legend(frameon=True,loc='upper right')
plt.savefig(output_path+'xgb_10_trklen_pid_BDTcut.eps')


#[11]B vs pid
plt.figure()
plt.figure(figsize = (12,9))
plt.title("Event Selection with BDT Cut") 
#plt.plot(x_ntrklen, y_pid,'ro', markersize=1, label='After event selection')

#plt.plot(XY_valid.loc[:, 'ntrklen'], XY_valid.loc[:, 'PID'], 'ko', markersize=1, label='All events')

plt.plot(XY_valid_inel[XY_valid_inel.tag==1].loc[:,'B'], XY_valid_inel[XY_valid_inel.tag==1].loc[:,'PID'], 'ro', markersize=1, label='Inel.')
plt.plot(XY_valid_inel[XY_valid_inel.tag==2].loc[:,'B'], XY_valid_inel[XY_valid_inel.tag==2].loc[:,'PID'], 'bo', color='blue', markersize=4, label='El.')
plt.plot(XY_valid_inel[XY_valid_inel.tag==5].loc[:,'B'], XY_valid_inel[XY_valid_inel.tag==5].loc[:,'PID'], 'go', color='green', markersize=4, label='MisID:P')

plt.xlabel("Impact Parameter [cm]")
plt.ylabel("$\chi^{2}$ PID [a.u.]")

plt.xlim(0, 20)
plt.ylim(-.5, 250)
plt.legend(frameon=True,loc='upper right')
plt.savefig(output_path+'xgb_11_B_PID_BDTcut.eps')

#save validation file -----------------------------
XY_valid_inel.to_csv(output_inel_csv, index=False)


