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
models = pd.DataFrame(columns=["Model","R2 Score"])
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

### KNN
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
new_row = {"Model": "KNN", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)


# LinearRegression
model_LR=LinearRegression()
scoring='neg_mean_squared_error'
results_LR=cross_val_score(model_LR,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_LR.mean())
model_LR.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_LR.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "LinearRegression", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#XGBoost
model_xgb=XGBRegressor(n_estimators=31, learning_rate=0.200879, max_depth=5)
results_xgb=cross_val_score(model_xgb,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_xgb.mean())
model_xgb.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_xgb.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "XGBoost", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#ElasticNetRegression

model_EN=ElasticNet()
results_EN=cross_val_score(model_EN,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_EN.mean())

model_EN.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_EN.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "ElasticNet", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#BaggingDecisionTreeRegressor
model_cart_bagging=DecisionTreeRegressor(max_depth=9)
model_bagging=BaggingRegressor(estimator=model_cart_bagging, n_estimators=10, random_state=76)
results_bagging=cross_val_score(model_bagging,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_bagging.mean())

model_bagging.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_bagging.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "BaggingDecisionTree", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#Ada Boost
model_ada=AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=9), n_estimators=40)
results_ada=cross_val_score(model_ada,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_ada.mean())

model_ada.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_ada.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "Ada Boost", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

# DecisionTreeRegressor
params_cart=dict(max_depth=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18])
model_cart=DecisionTreeRegressor()
grid_cart=GridSearchCV(estimator=model_cart, param_grid=params_cart, scoring=scoring, cv=kfold)
grid_cart.fit(x_selected, y_scaled)
print("Best Score: %f use parameters: %s" % (grid_cart.best_score_, grid_cart.best_params_))

r2 = r2_score(y_scaled, grid_cart.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "DecisionTree", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#SVM
model_svm=SVR(kernel='rbf', gamma=0.4, C=13)
results_svm=cross_val_score(model_svm,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_svm.mean())

model_svm.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_svm.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "SVM", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#GradientBoosting
model_gradientBoosting=GradientBoostingRegressor(n_estimators=54, learning_rate=0.10903, max_depth=5)
results_gradientBoosting=cross_val_score(model_gradientBoosting,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_gradientBoosting.mean())

model_gradientBoosting.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_gradientBoosting.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "GradientBoosting", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#RandomForest
model_randomforest=RandomForestRegressor()
params_rt=dict(max_depth=[7,8,9,10,11,12,13,14,15,16,18])
grid_rt=GridSearchCV(estimator=model_randomforest, param_grid=params_rt, scoring=scoring, cv=kfold, n_jobs=-1)
grid_rt.fit(x_selected, y_scaled.ravel())
print("Best Score: %f use parameters: %s" % (grid_rt.best_score_, grid_rt.best_params_))

r2 = r2_score(y_scaled, grid_rt.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "RandomForest", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#RidgeRegression
model_Ridge=Ridge()
results_Ridge=cross_val_score(model_Ridge,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_Ridge.mean())

model_Ridge.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_Ridge.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "RidgeRegression", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#ExtraTree

model_extratrees=ExtraTreesRegressor(max_depth=60)
results_extratrees=cross_val_score(model_extratrees,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_extratrees.mean())

model_extratrees.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_extratrees.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "ExtraTree", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

#LassoRegression
model_Lasso=Lasso()
results_Lasso=cross_val_score(model_Lasso,x_selected, y_scaled, cv=kfold, scoring=scoring)
print(results_Lasso.mean())

model_Lasso.fit(x_selected, y_scaled)
r2 = r2_score(y_scaled, model_Lasso.predict(x_selected))
print(f"R^2 Score: {r2}")
new_row = {"Model": "LassoRegression", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)

colors = sns.color_palette("husl", len(models))
#print bar chart
plt.figure(figsize=(12,8))
sns.barplot(x=models["Model"], y=models["R2 Score"])
plt.title("Models' R2 Score", size=15)
plt.xticks(rotation=30, size=12)
plt.show()
# Final model training with Gradient Boosting
final_gradient = GradientBoostingRegressor(learning_rate=0.10903, max_depth=5, n_estimators=54)
final_gradient.fit(x_selected, y_scaled)

# Load test data
test = pd.read_csv("C:/Users/Minh MPC/Downloads/test_file.csv")
print(test.head())

# Prepare test data
Id_pred = test['id']
Room_pred = test['room']
Area_pred = test['area']
Toilet_pred = test['toilet']
test.drop(['room', 'area', 'toilet', 'id'], inplace=True, axis=1)
test_en = pd.get_dummies(test, drop_first=True)
print(test_en.shape)

# Append test data to train data to align columns
result_test = x_en._append(test_en, sort=False)
print(result_test.shape)

# Select test data from the appended data
test_en_2 = result_test[2204:8813]
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
ypred_scale = final_gradient.predict(test_selected)
ypred_scale = pd.DataFrame(ypred_scale)
ypred = target_scaler.inverse_transform(ypred_scale)
print(ypred)

# Prepare prediction DataFrame
pred_data = pd.DataFrame(ypred, columns=['price'])
target_pred = pd.concat([Id_pred, Room_pred, Area_pred, Toilet_pred, pred_data], axis=1)
print(target_pred.head())

# Save predictions to CSV


# Calculate and print model score


# Calculate and print R^2 score
r2 = r2_score(y_scaled, final_gradient.predict(x_selected))
print(f"R^2 Score: {r2}")


new_row = {"Model": "final_gradient", "R2 Score": r2}
models = models._append(new_row, ignore_index=True)


