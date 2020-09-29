#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:11:34 2020

@author: eduardosolloa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training Linear regression model on the dataset

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

#Training polynomial regression on the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
"""Higher degrees makes it fit more"""
X_poly = poly_reg.fit_transform(X)
polynomial_regressor = LinearRegression()
polynomial_regressor.fit(X_poly, y)


#Visualizing Linear regression results

y_linear = linear_regressor.predict(X)
plt.figure()
plt.scatter(X, y, color="g")
plt.plot(X, y_linear, color="m", label="linear regression")

#Visualizing Polynomial regression results

y_poly = polynomial_regressor.predict(X_poly)
plt.plot(X, y_poly, color="c", label="polynomial regression")
plt.legend()

print(polynomial_regressor.predict(poly_reg.fit_transform([[6.5]])))