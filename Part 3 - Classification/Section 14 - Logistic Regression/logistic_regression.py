#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:47:34 2020

@author: eduardosolloa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Avoiding Dummy variable trap
X = X[:, 1:]

# Spliting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 1:] = sc_X.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc_X.transform(X_test[:, 1:])

# Fitting classifier to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting New Results
print(sc_X.transform([[30, 87000]]))
print(classifier.predict([[1, -0.21568634359976666, 2.146015658291428]]))
prob = classifier.predict_proba([[1, -0.21568634359976666, 2.146015658291428]])
print(f"The probability this client buys a car is: {prob[:, 1]}")

# Predicting the test results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis=1))

# Making the CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
# C00 is true negatives, C10 is false negatives, C11 is true positives, C01 false positives
print("Accuracy score:", accuracy_score(y_test, y_pred))
