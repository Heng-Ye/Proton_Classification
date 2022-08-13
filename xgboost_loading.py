import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

model_name="xgb.model"


model_xgb=xgb.Booster()
model_xgb.load_model(model_name)



