import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import plot_importance # Import the function

# importing necessary models
import numpy as np
import pandas as pd 
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score,roc_curve,auc
# ROC and AUC should be obtained on test set
# Suppose the ground truth is 'y_test', and the output score is named as 'y_score'



# Once the training is done, the plot_importance function can thus be used to plot the feature importance.
model_name="xgb.model"

#model_xgb=xgb.Booster()
model_xgb=XGBClassifier()
model_xgb.load_model(model_name)
plot_importance(model_xgb) # suppose the xgboost object is named "xgb"

plt.savefig("importance_plot.pdf") # plot_importance is based on matplotlib, so the plot can be saved use plt.savefig()

# ROC and AUC should be obtained on test set
# Suppose the ground truth is 'y_test', and the output score is named as 'y_score'


'''
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show() # display the figure when not using jupyter display
plt.savefig("roc.png") # resulting plot is shown below
'''

