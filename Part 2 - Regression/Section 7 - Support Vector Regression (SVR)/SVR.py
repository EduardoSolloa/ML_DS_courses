#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:19:23 2020

@author: eduardosolloa
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values  # ponemos [:, -1:] para que se cree un 2D array para poder hacer el feature scaling

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc2 = StandardScaler()
X = sc.fit_transform(X)
y = sc2.fit_transform(y)


#Training the SVR model on the whole dataset. Because it's small
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X, y)



#Predicting a new result
scaled_prediction = regressor.predict(sc.transform([[6.5]]))
prediction = sc2.inverse_transform(scaled_prediction)
print(prediction)

#Visualizing the SVR results
plt.figure()
plt.subplots_adjust(left=.16, bottom=.285)
ax = plt.subplot()
plt.scatter(sc.inverse_transform(X), sc2.inverse_transform(y), color="b", marker="x")
plt.plot(sc.inverse_transform(X), sc2.inverse_transform(regressor.predict(X)), color="c", label="SVR")
plt.ylabel("salary")
ax.set_xticks(range(1,11))
ax.set_xticklabels(dataset.iloc[:, 0].values, rotation=70)
plt.legend()
plt.show()


#Visualizing with a smoother curve
plt.figure()
X_grid = np.arange(min(sc.inverse_transform(X)), max(sc.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc.inverse_transform(X), sc2.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc2.inverse_transform(regressor.predict(sc.transform(X_grid))), color = 'c')
plt.show()