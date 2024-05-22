import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score, precision_score
import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier


data = pd.read_csv('/content/train_file.csv')
data.head(10)

# create the bins for the 13 classes
bins = [0,20,30, 45, 60,400]

# use the 'cut' function to divide the 'price' column into 13 classes
data['price_class'] = pd.cut(data['price'], bins, labels=False)
# you can now access the 'price_class' column to see which class each price belongs to

# Get the unique class labels
class_labels = data.price_class.unique()

# Create a list to store the oversampled dataframes
data_oversampled = []

# Loop over each class label
for label in class_labels:
    # Separate majority and minority classes
    data_majority = data[data.price_class!=label]
    data_minority = data[data.price_class==label]
 
    # Upsample minority class
    data_minority_upsampled = resample(data_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(data_majority),    # to match majority class
                                     random_state=123) # reproducible results
 
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    
    # Add the oversampled dataframe to the list
    data_oversampled.append(data_upsampled)

# Concatenate all the oversampled dataframes
data_oversampled = pd.concat(data_oversampled)

# Display new class counts
print(data_oversampled.price_class.value_counts())

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

# Create the Label Encoder
le = LabelEncoder()

# Fit and transform the non-numeric columns
data_oversampled['polistic'] = le.fit_transform(data_oversampled['polistic'])
data_oversampled['furniture'] = le.fit_transform(data_oversampled['furniture'])
data_oversampled['quan'] = le.fit_transform(data_oversampled['quan'])
data_oversampled['direct'] = le.fit_transform(data_oversampled['direct'])
data_oversampled['room'] = le.fit_transform(data_oversampled['room'])
data_oversampled['toilet'] = le.fit_transform(data_oversampled['toilet'])

# Create the feature and target arrays
X = data_oversampled[['room', 'toilet', 'area', 'x', 'y', 'quan', 'polistic', 'furniture', 'direct',  'n_hospital']] 
y = data_oversampled['price_class']

# Scale the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,y_train =X,y
test = pd.read_csv('/content/test_file.csv')
test.head(10)

# create the bins for the 13 classes
bins = [0,20,30, 45, 60,400]

# use the 'cut' function to divide the 'price' column into 13 classes
test['price_class'] = pd.cut(test['price'], bins, labels=False)
# you can now access the 'price_class' column to see which class each price belongs to

# Get the unique class labels
class_labels = test.price_class.unique()

# Create a list to store the oversampled dataframes
test_oversampled = []

# Loop over each class label
for label in class_labels:
    # Separate majority and minority classes
    test_majority = test[test.price_class!=label]
    test_minority = test[test.price_class==label]
 
    # Upsample minority class
    test_minority_upsampled = resample(test_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(test_majority),    # to match majority class
                                     random_state=123) # reproducible results
 
    # Combine majority class with upsampled minority class
    test_upsampled = pd.concat([test_majority, test_minority_upsampled])
    
    # Add the oversampled dataframe to the list
    test_oversampled.append(test_upsampled)

# Concatenate all the oversampled dataframes
test_oversampled = pd.concat(test_oversampled)

# Display new class counts
print(test_oversampled.price_class.value_counts())
test_oversampled['polistic'] = le.fit_transform(test_oversampled['polistic'])
test_oversampled['furniture'] = le.fit_transform(test_oversampled['furniture'])
test_oversampled['quan'] = le.fit_transform(test_oversampled['quan'])
test_oversampled['direct'] = le.fit_transform(test_oversampled['direct'])
test_oversampled['room'] = le.fit_transform(test_oversampled['room'])
test_oversampled['toilet'] = le.fit_transform(test_oversampled['toilet'])

# Create the feature and target arrays
X_t = test_oversampled[['room', 'toilet', 'area', 'x', 'y', 'quan', 'polistic', 'furniture', 'direct',  'n_hospital']] 
y_t = test_oversampled['price_class']

# Scale the feature data
scaler = StandardScaler()
X_t = scaler.fit_transform(X_t)

# Split the data into training and test sets
X_test,y_test =X_t,y_t
knn = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
knn.fit(X_train, y_train)
# neighbors = np.arange(1, 31)
# train_accuracy = np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))

# # Create a KNN classifier with Euclidean metric
# for i, k in enumerate(neighbors):
#     knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#     knn.fit(X_train, y_train)
    
#     test_accuracy[i] = knn.score(X_test, y_test)

# plt.plot(neighbors, test_accuracy, label='Euclidean')

# # Create a KNN classifier with Manhattan metric
# for i, k in enumerate(neighbors):
#     knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
#     knn.fit(X_train, y_train)
    
#     test_accuracy[i] = knn.score(X_test, y_test)

# plt.plot(neighbors, test_accuracy, label='Manhattan')

# plt.xlabel('Number of neighbors')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Create an Extra Trees Classifier model
# model = ExtraTreesClassifier(n_estimators=100)
# model.fit(X_train, y_train)

# # Get feature importances
# importances = model.feature_importances_

# # Print the feature importances
# print("Feature Importances:")
# features = ['n_rooms','n_toilets','area','x','y','distance_UBND','district','polistic','furniture','house_direct','balcony_direct','n_schools','n_hospitals']
# for feature, importance in zip(features, importances):
#     print(f'{feature}: {importance}')

# evaluate the model 
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

# make predictions
y_pred = knn.predict(X_test)


# calculate the F1 score,recall,precision and AUC
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')


print('Train Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)
print("F1 Score: ", f1)
print("Recall: ", recall)
print("Precision: ", precision)

##save to csv file
from sklearn.preprocessing import MinMaxScaler
target_scaler=MinMaxScaler()
y_data=pd.DataFrame(y_train)
target_scaler.fit(y_data)
ypred_scale=y_pred
ypred_scale=pd.DataFrame(ypred_scale)
ypred=target_scaler.inverse_transform(ypred_scale)
ypred

Id_pred=test['id']
Room_pred=test['room']
Area_pred=test['area']
Toilet_pred=test['toilet']
x_pred=test['x']
y_pred=test['y']
khoang_cach_pred=test['khoang_cach']
n_hospital_pred=test['n_hospital']
pred_data=pd.DataFrame(ypred,columns=['price_class'])
target_pred=pd.concat([Id_pred,Room_pred,Area_pred,Toilet_pred,x_pred,y_pred,khoang_cach_pred,n_hospital_pred,pred_data],axis=1)
target_pred.head()

target_pred

target_pred.to_csv("/content/KNN.csv", index=False)

