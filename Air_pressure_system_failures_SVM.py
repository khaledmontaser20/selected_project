# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 08:43:59 2021

@author: DELL
"""

# Importing Needed Libarary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------

# Read Data Frame
path_training = "C:\\Users\\DELL\\Documents\\Selected_Projects\\SVM\\Numrical SVM Project\\aps_failure_training_set.csv"
path_testing = "C:\\Users\\DELL\\Documents\\Selected_Projects\\SVM\\Numrical SVM Project\\aps_failure_test_set.csv"
dataset_training = pd.read_csv(path_training)
dataset_testing = pd.read_csv(path_testing)

# ----------------------------------------------------------------------------

# # Show Training Data Sample
# print(dataset_training.head(10))
# print("=*=" * 30)
# print(dataset_training.tail(10))
# print("=*=" * 30)
# print("=*=" * 30)
# # Show Testing Data Sample
# print(dataset_testing.head(10))
# print("=*=" * 30)
# print(dataset_testing.tail(10))

# ----------------------------------------------------------------------------

# # Know Columns Data Types Of Training Data
# print(dataset_training.dtypes)

# # Know Columns Data Types Of Testing Data
# print(dataset_testing.dtypes)

# # Observe That 170 Columns Has Null Values

# ----------------------------------------------------------------------------

# # Training Data Set Info
# print(dataset_training.info())
# print("=*=" * 30)
# # Testing Data Set Info
# print(dataset_testing.info())

# ----------------------------------------------------------------------------

# # Get Training Data Set Columns
# print(dataset_training.columns)
# print("=*=" * 30)
# # Get Testing Data Set Columns
# print(dataset_testing.columns)

# ----------------------------------------------------------------------------

# # Show Unique Values in Class Column in Training Data set
# print(dataset_training['class'].unique())
# print("=*=" * 30)
# # Show Unique Values in Class Column in Testing Data set
# print(dataset_testing['class'].unique())

# ----------------------------------------------------------------------------

# Replacing neg and pos in the class column with the appropriate Integer Values 0s and 1s in Training Data
dataset_training['class'] = dataset_training['class'].replace('neg', 0)
dataset_training['class'] = dataset_training['class'].replace('pos', 1)

# Replacing neg and pos in the class column with the appropriate Integer Values 0s and 1s in Testing Data
dataset_testing['class'] = dataset_testing['class'].replace('neg', 0)
dataset_testing['class'] = dataset_testing['class'].replace('pos', 1)

# # Show Unique Values in Class Column in Training Data set
# print(dataset_training['class'].unique())
# print("=*=" * 30)
# # Show Unique Values in Class Column in Testing Data set
# print(dataset_testing['class'].unique())

# ----------------------------------------------------------------------------

# Function Calculate Mean
def get_mean(obj):
    sum = 0
    for raw in obj:
        try:
           raw = int(raw)
#          print(type(raw), raw)
           sum += raw 
        except Exception:
            pass
    return sum // len(obj)

# print(get_mean(dataset_training['ab_000']))

# ----------------------------------------------------------------------------

# Data Rescalling
def data_rescalling(dataset):
    # dataset = (dataset - dataset.mean()) / dataset.std()
    for column in dataset.columns:
        dataset[column] = ( ( dataset[column] - get_mean(dataset[column]) ) / dataset[column].std() )
    # print("Data After Rescalling (Normalization)\n", dataset.head(10))

# ----------------------------------------------------------------------------

# Function Calculate Median
def get_median(obj):
    numbers = set()
    count = 0
    for raw in obj:
        try:
           raw = int(raw)
           #print(type(raw), raw)
           numbers.add(raw)
           #print(numbers)
        except Exception:
            pass
    numbers= list(numbers)
    #print(type(numbers), numbers)
    return numbers[len(numbers)//2]

# ----------------------------------------------------------------------------

def data_rescalling2(dataset):
    for column in dataset.columns:
        if dataset[column].min() == dataset[column].max():
            dataset[column] = 0
        else:
            dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())

# ----------------------------------------------------------------------------

# Replace na with numpy.nan to fillna with median value (Because fillna not support na)
for column in dataset_training.columns:
    dataset_training[column] = dataset_training[column].replace('na', np.nan)
    dataset_testing[column] = dataset_testing[column].replace('na', np.nan)
    #print(dataset_training[column].unique())

# ----------------------------------------------------------------------------

# Replaceing NULL Values By mean() Or Median()
#count = 0
for column in dataset_training.columns:
    median_value = get_median(dataset_training[column])
    #print(median_value)
    #count +=1
    dataset_training[column].fillna(value=median_value, inplace=True)
    dataset_testing[column].fillna(value=median_value, inplace=True)
# print("+="*20)
# print(count)
# print("+="*20)

# # For Make Sure That In 171 Columns No Na Or NaN or Null Values Exist 
# for column in dataset_training.columns:
#       print(dataset_training[column].unique())
#       print("\n\n============================================================")
     
# ----------------------------------------------------------------------------

# Convert Columns Of Object Data Type To Int
# convert type of object to int in ml python
for column in dataset_training.columns:
    try:
        # dataset_training[column] = dataset_training[column].astype('int64')
        # dataset_testing[column] = dataset_testing[column].astype('int64')
        
        dataset_training[column] = dataset_training[column].astype('float64')
        dataset_testing[column] = dataset_testing[column].astype('float64')
    except ValueError:
        print(column)
    
# # Check If Converted Succesfully
# print(dataset_training.dtypes)
# print(dataset_testing.dtypes)

# ----------------------------------------------------------------------------

# Split Data Set To X, y In Training and Testing
cols = dataset_training.shape[1]
X_training = dataset_training.iloc[:, 1:cols]
y_training = dataset_training.iloc[:, 0:1]

X_testing = dataset_testing.iloc[:, 1:cols]
y_testing = dataset_testing.iloc[:, 0:1]

# ----------------------------------------------------------------------------

# Data Rescaling _ Calling To Rescalling Function
data_rescalling2(X_training)
data_rescalling2(X_testing)

# count = 0
# for raw in X_training['cd_000']:
#     # print(raw, '_')
#     try:
#         if int(raw) == 1209600: # 60_000 - 59324 = 00676
#             count +=1
#     except Exception:
#         pass
# # print("count: ", count)

# ----------------------------------------------------------------------------

# Insert New Columns by Values 1 as X0
# Ob.insert(pos, name, value)
X_training.insert(0, 'Ones', 1)
# print('New data = \n', X_training.head(10))

X_testing.insert(0, 'Ones', 1)
# print('New data = \n', X_testing.head(10))

# ----------------------------------------------------------------------------

# # -------------------------------- SVM CODE --------------------------------

# svc = svm.SVC()
# Instantiate the Support Vector Classifier (SVC)
svc = svm.SVC(C=1.0, random_state=1, kernel='linear')
print(X_training.shape, y_training.shape, X_testing.shape, y_testing.shape)

# ----------------------------- TRIANING SVM CODE ----------------------------

svc.fit(X_training, y_training)
 
# Fit the model
svc.fit(X_training, y_training)

# ------------------------------ TESTING SVM CODE ----------------------------

print("Test accuracy = {0}%".format(np.round(svc.score(X_testing, y_testing) * 100), 2))
y_predicted = svc.predict(X_testing)
yPredicted = list()
yTesting = list()

for i in y_predicted:
    #print(i)
    yPredicted.append(i)
    
for i in y_testing:
    #print(i)
    yTesting.append(i)
    
# yPredicted = yPredicted.replace(0, 'neg')
# yPredicted = yPredicted.replace(1, 'pos')

for i in range(len(yPredicted)-1):
    if yPredicted[i] == 0:
       yPredicted[i] = 'neg'
    elif yPredicted[i] == 1:
        yPredicted[i] = 'pos'
    else:
        pass
    
for i in range(len(yTesting)-1):
    if yTesting[i] == 0:
       yTesting[i] = 'neg'
    elif yTesting[i] == 1:
        yTesting[i] = 'pos'
    else:
        pass


for i in range(10):
    print(yPredicted[i])
    
# for i in range(10):
#     print(yTesting[i])



# ----------------------------------------------------------------------------