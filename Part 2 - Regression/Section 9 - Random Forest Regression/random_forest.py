#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:46:42 2020

@author: eduardosolloa
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=150)
regressor.fit(X, y)

#Predicting a new result
print(regressor.predict([[6.5]]))

#Visualizing the random forest regression results (high res)
x_hd = np.arange(min(X), max(X), .1)
x_hd = x_hd.reshape(len(x_hd), 1)
plt.subplots_adjust(left=.17)
plt.scatter(X, y, color="g")
plt.plot(x_hd, regressor.predict(x_hd), color="orange", label="tree predictions")
plt.ylabel("salary")
plt.xlabel("position level")
plt.legend()
plt.show()

# R^2 
#from skelearn.metrics import r2_score
#r2_score(y_test, y_pred)