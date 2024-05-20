import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import os
data=pd.read_csv("C:/Users/Minh MPC/Downloads/train_file.csv")
data.head(10)
data[data.columns[data.isna().sum() > 0]].isna().mean()*100
X=data.drop('price', axis=1)
y=data['price']
x_en=pd.get_dummies(X,drop_first=True)
imputer=KNNImputer()
imputer.fit(x_en)
x=imputer.transform(x_en)
x=pd.DataFrame(x, columns=x_en.columns)
x[x.columns[x.isna().sum() > 0]].isna().mean()*100
lof = LocalOutlierFactor()
yhat = lof.fit_predict(x.to_numpy())
mask=yhat!=-1
x_train=x.to_numpy()[mask, :]
x_train=pd.DataFrame(x_train, columns=x.columns)
y_train=y[mask]
print(x_train.shape, y_train.shape)
mm_scaler=MinMaxScaler()
x_scaled=pd.DataFrame(mm_scaler.fit_transform(x_train), columns=x_train.columns)
x_scaled.head()
target_scaler=MinMaxScaler()
y_data=pd.DataFrame(y_train)
target_scaler.fit(y_data)
y_scaled=target_scaler.transform(y_data)
y_scaled
dtr=DecisionTreeRegressor()
dtr.fit(x_scaled,y_scaled)
rfe=RFE(dtr)
rfe=rfe.fit(x_scaled,y_scaled)
x_scaled.columns[rfe.support_]
selected_list=['room', 'area', 'x', 'y', 'khoang_cach',
       'n_hospital']
x_selected=x_scaled[selected_list]
x_selected.head()
x_features=x_selected.columns
x_features



scoring='neg_mean_squared_error'
kfold=KFold(n_splits=10, random_state=76, shuffle=True)


model_EN=ElasticNet()
results_EN=cross_val_score(model_EN,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_EN.mean())
model_EN.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_EN.predict(x_selected))
print(f"R^2 Score: {r2}")

test=pd.read_csv("C:/Users/Minh MPC/Downloads/test_file.csv")
test.head()

Id_pred=test['id']
Room_pred=test['room']
Area_pred=test['area']
Toilet_pred=test['toilet']
x_pred=test['x']
y_pred=test['y']
khoang_cach_pred=test['khoang_cach']
n_hospital_pred=test['n_hospital']
test.drop(['room','area','toilet','id','x', 'y', 'khoang_cach',
       'n_hospital'], inplace=True, axis=1)
test_en=pd.get_dummies(test,drop_first=True)
print(test_en.shape)

result_test= x_en._append(test_en, sort=False)
result_test.shape

test_en_2=result_test[2204:4407]
test_en_2.shape
test_en_2[test_en_2.columns[test_en_2.isna().sum() > 0]].isna().mean()*100

test_en_3=imputer.transform(test_en_2)

test_en_3=pd.DataFrame(test_en_3, columns=test_en_2.columns)
test_en_3[test_en_3.columns[test_en_3.isna().sum() > 0]].isna().mean()*100

test_scaled = pd.DataFrame(mm_scaler.transform(test_en_3), columns=test_en_3.columns)
test_scaled.head()

test_selected=test_scaled[x_features]
test_selected.head()

ypred_scale=model_EN.predict(test_selected)
ypred_scale=pd.DataFrame(ypred_scale)
ypred=target_scaler.inverse_transform(ypred_scale)
ypred

pred_data=pd.DataFrame(ypred,columns=['price'])
target_pred=pd.concat([Id_pred,Room_pred,Area_pred,Toilet_pred,x_pred,y_pred,khoang_cach_pred,n_hospital_pred,pred_data],axis=1)
target_pred.head()

target_pred

target_pred.to_csv("C:/Users/Minh MPC/Downloads/ElasticNetRegression.csv", index=False)
