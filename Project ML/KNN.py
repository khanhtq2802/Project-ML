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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

# Load data
data = pd.read_csv("C:/Users/Minh MPC/Downloads/train_file.csv")
print(data.head(10))

# Check missing values
print(data[data.columns[data.isna().sum() > 0]].isna().mean() * 100)

# Split data into features and target
X = data.drop('price', axis=1)
y = data['price']

# One-hot encoding
x_en = pd.get_dummies(X, drop_first=True)

# Impute missing values
imputer = KNNImputer()
imputer.fit(x_en)
x = imputer.transform(x_en)
x = pd.DataFrame(x, columns=x_en.columns)
print(x[x.columns[x.isna().sum() > 0]].isna().mean() * 100)

# Outlier detection
lof = LocalOutlierFactor()
yhat = lof.fit_predict(x.to_numpy())
mask = yhat != -1
x_train = x.to_numpy()[mask, :]
x_train = pd.DataFrame(x_train, columns=x.columns)
y_train = y[mask]
print(x_train.shape, y_train.shape)

# Scaling features
mm_scaler = MinMaxScaler()
x_scaled = pd.DataFrame(mm_scaler.fit_transform(x_train), columns=x_train.columns)
print(x_scaled.head())

# Scaling target
target_scaler = MinMaxScaler()
y_data = pd.DataFrame(y_train)
target_scaler.fit(y_data)
y_scaled = target_scaler.transform(y_data)
print(y_scaled)

# Feature selection using RFE
dtr = DecisionTreeRegressor()
dtr.fit(x_scaled, y_scaled)
rfe = RFE(dtr)
rfe = rfe.fit(x_scaled, y_scaled)
print(x_scaled.columns[rfe.support_])

# Selected features
selected_list = ['room', 'area', 'x', 'y', 'khoang_cach', 'n_hospital']
x_selected = x_scaled[selected_list]
print(x_selected.head())
x_features = x_selected.columns
print(x_features)

# Grid search for KNN
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits=10, random_state=76, shuffle=True)
params_KNN = dict(n_neighbors=[i for i in range(15, 39)])
model_KNN = KNeighborsRegressor()
grid_KNN = GridSearchCV(estimator=model_KNN, param_grid=params_KNN, scoring=scoring, cv=kfold)
grid_KNN.fit(x_selected, y_scaled)
print("Best Score: %f use parameters: %s" % (grid_KNN.best_score_, grid_KNN.best_params_))

r2 = r2_score(y_scaled, grid_KNN.predict(x_selected))
print(f"R^2 Score: {r2}")

# Load test data
test = pd.read_csv("C:/Users/Minh MPC/Downloads/test_file.csv")
print(test.head())

# Prepare test data
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
test_en = pd.get_dummies(test, drop_first=True)
print(test_en.shape)

# Append test data to train data to align columns
result_test = x_en._append(test_en, sort=False)
print(result_test.shape)

# Select test data from the appended data
test_en_2 = result_test[2204:4407]
print(test_en_2.shape)
print(test_en_2[test_en_2.columns[test_en_2.isna().sum() > 0]].isna().mean() * 100)

# Impute missing values in test data
test_en_3 = imputer.transform(test_en_2)
test_en_3 = pd.DataFrame(test_en_3, columns=test_en_2.columns)
print(test_en_3[test_en_3.columns[test_en_3.isna().sum() > 0]].isna().mean() * 100)

# Scale test data
test_scaled = pd.DataFrame(mm_scaler.transform(test_en_3), columns=test_en_3.columns)
print(test_scaled.head())

# Select features for prediction
test_selected = test_scaled[x_features]
print(test_selected.head())

# Predict using the final model
ypred_scale = grid_KNN.predict(test_selected)
ypred_scale = pd.DataFrame(ypred_scale)
ypred = target_scaler.inverse_transform(ypred_scale)
print(ypred)

# Prepare prediction DataFrame
pred_data = pd.DataFrame(ypred, columns=['price'])
target_pred=pd.concat([Id_pred,Room_pred,Area_pred,Toilet_pred,x_pred,y_pred,khoang_cach_pred,n_hospital_pred,pred_data],axis=1)
print(target_pred.head())

# Save predictions to CSV
target_pred.to_csv("C:/Users/Minh MPC/Downloads/KNN.csv", index=False)


