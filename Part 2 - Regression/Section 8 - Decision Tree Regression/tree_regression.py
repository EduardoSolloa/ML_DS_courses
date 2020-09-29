#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:16:50 2020

@author: eduardosolloa
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""There is no feature scaling needed in desicion tree regressions"""

#Training the decision tree regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting a new result
print(regressor.predict([[6.5]]))

#Visualizing the decision tree regression results (high res)
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