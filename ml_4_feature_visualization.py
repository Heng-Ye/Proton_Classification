# importing necessary models -----------------------------------------------
from array import array
import numpy as np
import pandas as pd 
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)

from babyplots import Babyplot

import xgboost as xgb
from xgboost import XGBRegressor # Or XGBClassifier for Classification
from xgboost import plot_tree


# Sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#t-SNE
from sklearn.manifold import TSNE

# UMAP
import umap.umap_ as umap
from umap import UMAP

import plotly.express as px

import mplhep as hep

#HEP- Plot Style ------------------------------
hep.style.use("CMS") # string aliases work too
#hep.style.use(hep.style.ROOT)
#hep.style.use(hep.style.ATLAS)

#Read training and validation files ----------------------------------------
input_file_name='./csv/xgb_inel_valid'
output_file_name='./models/lgbm.model'
output_path='./plts_perform/'

train_data = pd.read_csv(input_file_name+'.csv') #read input file
#print(train_data.head(3))
#var_colums=[c for c in train_data.columns if c not in ['train','tag','target']]
var_colums=[c for c in train_data.columns if c not in ['bdt_score','train','target']]
X=train_data.loc[:, var_colums]

#Read feature observables --------------------------------------------------------
#feature_names=[c for c in X_train.columns if c not in ['bdt_score','train','tag','target']]

#Add tag color in dataframe ----------------------------------
#print('size of data:',len(X))
tag_color=[]
tag_label=[]
for index, row in X.iterrows():
    tag=row["tag"]
    color="snow"
    label="empty"
    #color_bool=0
    if tag == 1: 
        color="red"
        label="Inel"
    if tag == 2: 
        color="blue"
        label="El"
    if tag == 3: 
        color="yellow"
        label="MisID:cosmic"
    if tag == 4: 
        color="fuchsia"
        label="MisID:pi"
    if tag == 5: 
        color="green"
        label="MisID:P"
    if tag == 6: 
        color="sienna"
        label="MisID:mu"
    if tag == 7: 
        color="mediumseagreen"
        label="MisID:e/gamma"
    if tag == 8: 
        color="silver"
        label="MisID:other"
    tag_color.append(color)
    tag_label.append(label)
    #print('[',index,'] tag=',tag,' color:',color)

XL=X.copy()
XL.insert(0, "color", tag_color, True)
XL.insert(1, "label", tag_label, True)
#print(XL.head(15))



#2D Projection with UMAP ----------------------------------------
'''
features = X.loc[:, :'PID'] 

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(features)

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=tag_color, labels=tag_color
    #color=df.species, labels={'color': 'species'}
)
fig_2d.show()
'''

#3D Projection with UMAP ----------------------------------------
features = X.loc[:, :'PID'] 
#features = X.loc[:, :'ntrklen'] 
#features = X.loc[:, :'B'] 
#features = X.loc[:, :'B'] 
#features = X.loc[:, :'endpointdedx'] 
'''
umap_3d = UMAP(n_components=3, init='random', random_state=1)
proj_3d = umap_3d.fit_transform(features)
fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=tag_color
)
fig_3d.update_traces(marker_size=5)

fig_3d.show()
'''

#FIXME: Bug in tag color


umap_3d = UMAP(n_components=3)
components_umap = umap_3d.fit_transform(X)
# 3D scatterplot
fig = px.scatter_3d(
    components_umap, x=0, y=1, z=2, color=tag_color, size=0.1*np.ones(len(X)), opacity = 1,
    title='UMAP plot in 3D',
    #labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=1200, height=900
)
fig.show()







'''
umap_2d = UMAP(random_state=0)
umap_2d.fit(features)

projections = umap_2d.transform(features)
fig = px.scatter(
    projections, x=0, y=1,
    color=tag_color, labels=tag_label
)
fig.show()
'''