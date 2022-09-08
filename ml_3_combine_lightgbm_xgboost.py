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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
#output_file_name='./models/lgbm.model'
#output_path='./plts_perform/'
#output_path='./plts_perform_20220902/'
#input_file_name='./data_prep/protons_mva2_20220902'
#output_file_name='./models/lgbm_mva2_20220902.model'

#Read file, feature observables, output file ------------------------------------------------------
parser = ap()
parser.add_argument("-d", type=str, help='Data File', default = "")
parser.add_argument("-f", type=str, help='Feature variables:', default = "'train','tag','target'")
parser.add_argument("-o1", type=str, help='input model name1(lightgbm):', default = "")
parser.add_argument("-o2", type=str, help='input model name2(xgb):', default = "")
parser.add_argument("-p", type=str, help='Output plot path:', default = "")
parser.add_argument("-ocsv", type=str, help='Output csv file:', default = "")


args=parser.parse_args()
if not (args.d and args.f and args.o1 and args.o2):
  print("--> Please provide input data file, feature observables, and output model name")
  exit()

input_file_name=args.d
feature_obs='['+args.f+']'
output_file_name1=args.o1
output_file_name2=args.o2
output_path=args.p
output_inel_csv=args.ocsv


X_train=pd.read_csv(input_file_name+'_X_train.csv')
X_valid=pd.read_csv(input_file_name+'_X_valid.csv')
y_train=pd.read_csv(input_file_name+'_y_train.csv')
y_valid=pd.read_csv(input_file_name+'_y_valid.csv')
z_valid=pd.read_csv(input_file_name+'_z_valid.csv')

#Read feature observables --------------------------------------------------------
#feature_names=[c for c in X_train.columns if c not in ['train','tag','target']]
feature_names=[c for c in X_train.columns if c not in feature_obs]

#[1] Load models -----------------------------------------
#lgbm
model_lgbm=lgbm.Booster(model_file=output_file_name1)
model_lgbm.feature_name = feature_names
y_pred_lgbm = model_lgbm.predict(X_valid)

#xgboost
model_xgb=xgb.XGBRegressor()
model_xgb.load_model(output_file_name2)
model_xgb.get_booster().feature_names = feature_names
y_pred_xgb = model_xgb.predict(X_valid)

#[2] Merge all info for valid set ------------------
XY_valid=X_valid.copy()
XY_valid.insert(0, "bdt_lgbm", y_pred_lgbm, True)
XY_valid.insert(1, "bdt_xgb", y_pred_xgb, True)
XY_valid.insert(2, "target", y_valid, True)
XY_valid.insert(3, "tag", z_valid, True)

#new_score=(XY_valid['bdt_lgbm']+XY_valid['bdt_xgb'])/np.sqrt(2) #small improvement
new_score=(XY_valid['bdt_lgbm']+XY_valid['bdt_xgb'])/2.
#new_score=np.sqrt((XY_valid['bdt_lgbm']*XY_valid['bdt_xgb']))
XY_valid.insert(2,"bdt_com", new_score, True)


print(y_valid.head(3))
print(XY_valid.head(3))
print('Dimensions of y_valid:',y_valid.shape)
print('Dimensions of XY_valid:',XY_valid.shape)


#[3]2D scatter plot ---------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize = (12,9))
plt.title("") 
plt.plot(XY_valid[XY_valid.tag==1].loc[:,'bdt_lgbm'], XY_valid[XY_valid.tag==1].loc[:,'bdt_xgb'], 'ro', markersize=4, label='Inel.')
plt.plot(XY_valid[XY_valid.tag==2].loc[:,'bdt_lgbm'], XY_valid[XY_valid.tag==2].loc[:,'bdt_xgb'], 'bo', color='blue', markersize=4, label='El.')
plt.plot(XY_valid[XY_valid.tag==5].loc[:,'bdt_lgbm'], XY_valid[XY_valid.tag==5].loc[:,'bdt_xgb'], 'go', color='green', markersize=4, label='MisID:P')

plt.xlabel("LightGBM BDT Score [a.u.]")
plt.ylabel("XGBoost BDT Score [a.u.]")

# Define Axes
X = [0,1]
Y = [0,1]
 
# Plot a graph
plt.plot(X, Y, linestyle='--', color='black')

#plt.xlim(0, 1.3)
#plt.ylim(-.5, 250)
plt.legend(frameon=True,loc='upper left')
plt.savefig(output_path+'1_BDT_scatter_plots.eps')


#[4] plot BDT score ----- ---------------------------------------------------------------------------------------------- 
plt.figure()
n_bdt=130
bdt_min=-.1
bdt_max=1.2
plt.figure(figsize = (12,9))
plt.title("Combined Score") 
plt.hist(XY_valid['bdt_com'],bins=np.linspace(bdt_min,bdt_max,n_bdt),histtype='step',color='black',label='All events', log=True)

# make the plot readable
#plt.xlabel('BDT Score',fontsize=12)
#plt.ylabel('Events',fontsize=12)
#plt.xlabel('(XGB+LGBM)/sqrt(2)', loc='center')
plt.xlabel('(XGB Score + LGBM Score)/2', loc='center')
plt.ylabel('Events', loc='center')
plt.legend(frameon=False)
#plt.savefig('BDT_predict.png')

# plot signal and background separately
#plt.figure()
#select true signal and background with 'target' label 1 and 0
bdt_strue=XY_valid.loc[XY_valid['target']==1]
bdt_btrue=XY_valid.loc[XY_valid['target']==0]

#select only the bdt_score column
bdt_s=bdt_strue.loc[:, 'bdt_com']
bdt_b=bdt_btrue.loc[:, 'bdt_com']

#print('Dimensions of bdt_strue:',bdt_strue.shape)
plt.hist(bdt_s,bins=np.linspace(bdt_min,bdt_max,n_bdt),histtype='step',color='red',label='signal', log=True)
plt.hist(bdt_b,bins=np.linspace(bdt_min,bdt_max,n_bdt),histtype='step',color='blue',label='background', log=True)

# make the plot readable
#plt.xlabel('Prediction from BDT',fontsize=12)
#plt.ylabel('Events',fontsize=12)
plt.legend(loc='upper left', frameon=False)
plt.savefig(output_path+'2_combined_BDTscore_dist.eps')


#evt selection cuts -------------------------------------------------------
bdt_cut=0.55
#select inel. events based on the BDT_score cut (in DataFrame)
XY_valid_inel=XY_valid.loc[XY_valid['bdt_com']>bdt_cut]
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

#[7]BDT cut ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
n_trklen=65
trklen_min=0
trklen_max=130
bins_each=np.linspace(trklen_min,trklen_max,n_trklen)

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
plt.savefig('lgbm_3_ntrklen_pid_inelcut.eps')
 

plt.figure(figsize = (12,9))
plt.title("Basic Cuts") 
plt.plot(XY_valid.loc[:, 'ntrklen'], XY_valid.loc[:, 'PID'], 'ko', markersize=1, label='All events')
plt.xlim(0, 1.3)
plt.ylim(-.5, 250)

plt.xlabel("Normalized Track Length [a.u.]")
plt.ylabel("$\chi^{2}$ PID [a.u.]")
plt.legend(frameon=True)
plt.savefig(output_path+'lgbm_8_ntrklen_pid_nocut.eps')

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
plt.savefig(output_path+'com_9_ntrklen_pid_BDTcut.eps')

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
plt.savefig(output_path+'com_10_trklen_pid_BDTcut.eps')


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
plt.savefig(output_path+'com_11_B_PID_BDTcut.eps')


