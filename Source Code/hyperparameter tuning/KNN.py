import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load training data
train_data = pd.read_csv("Training Data/regression/train.csv")
train_data.head(10)

# Create the bins for the classes
price_bins = [0, 20, 30, 45, 60, 400]

# Divide the 'price' column into classes
train_data['price_class'] = pd.cut(train_data['price'], price_bins, labels=False)

# Get the unique class labels
train_class_labels = train_data.price_class.unique()

# Create a list to store the oversampled dataframes
train_data_oversampled_list = []

# Loop over each class label
for label in train_class_labels:
    # Separate majority and minority classes
    train_data_majority = train_data[train_data.price_class != label]
    train_data_minority = train_data[train_data.price_class == label]

    # Upsample minority class
    train_data_minority_upsampled = resample(train_data_minority, 
                                             replace=True,     # sample with replacement
                                             n_samples=len(train_data_majority),    # to match majority class
                                             random_state=123) 

    # Combine majority class with upsampled minority class
    train_data_upsampled = pd.concat([train_data_majority, train_data_minority_upsampled])
    
    # Add the oversampled dataframe to the list
    train_data_oversampled_list.append(train_data_upsampled)

# Concatenate all the oversampled dataframes
train_data_oversampled = pd.concat(train_data_oversampled_list)

# Display new class counts
print(train_data_oversampled.price_class.value_counts())

# Encode categorical variables
label_encoder = LabelEncoder()
# Encode categorical variables using a for loop
categorical_columns = ['polistic', 'furniture', 'quan', 'direct', 'room', 'toilet']

for col in categorical_columns:
    train_data_oversampled[col] = label_encoder.fit_transform(train_data_oversampled[col])


# Create the feature and target arrays
X_train_data = train_data_oversampled[['room', 'toilet', 'area', 'x', 'y', 'quan', 'polistic', 'furniture', 'direct', 'n_hospital']] 
y_train_data = train_data_oversampled['price_class']

# Scale the feature data
scaler = StandardScaler()
X_train_data = scaler.fit_transform(X_train_data)

# Split the data into training and test sets
X_train, y_train = X_train_data, y_train_data

# Initialize and train the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
knn.fit(X_train, y_train)

# Load validation data
validation_data = pd.read_csv("Training Data/regression/val.csv")
validation_data['price_class'] = pd.cut(validation_data['price'], price_bins, labels=False)

# Get the unique class labels
validation_class_labels = validation_data.price_class.unique()

# Create a list to store the oversampled dataframes
validation_data_oversampled_list = []

# Loop over each class label
for label in validation_class_labels:
    # Separate majority and minority classes
    validation_data_majority = validation_data[validation_data.price_class != label]
    validation_data_minority = validation_data[validation_data.price_class == label]

    # Upsample minority class
    validation_data_minority_upsampled = resample(validation_data_minority, 
                                                  replace=True,     # sample with replacement
                                                  n_samples=len(validation_data_majority),    # to match majority class
                                                  random_state=123) 

    # Combine majority class with upsampled minority class
    validation_data_upsampled = pd.concat([validation_data_majority, validation_data_minority_upsampled])
    
    # Add the oversampled dataframe to the list
    validation_data_oversampled_list.append(validation_data_upsampled)

# Concatenate all the oversampled dataframes
validation_data_oversampled = pd.concat(validation_data_oversampled_list)

# Display new class counts
print(validation_data_oversampled.price_class.value_counts())

# Encode categorical variables
# Encode categorical variables for validation data using a for loop
for col in categorical_columns:
    validation_data_oversampled[col] = label_encoder.fit_transform(validation_data_oversampled[col])


# Create the feature and target arrays
X_validation_data = validation_data_oversampled[['room', 'toilet', 'area', 'x', 'y', 'quan', 'polistic', 'furniture', 'direct', 'n_hospital']] 
y_validation_data = validation_data_oversampled['price_class']

# Scale the feature data
X_validation_data = scaler.fit_transform(X_validation_data)

# Assign validation data
X_val, y_val = X_validation_data, y_validation_data

# Hyperparameter tuning
param_grid_knn = {
    'n_neighbors': [i for i in range(1, 10)],
    'p' : [1, 2]
}

scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits=10, random_state=76, shuffle=True)

grid_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring=scoring, cv=kfold)
grid_knn.fit(X_val, y_val)

print("Best Score: %f use parameters: %s" % (grid_knn.best_score_, grid_knn.best_params_))
