# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 18:05:26 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LinearRegression
import matplotlib.pyplot as plt
hausing=fetch_california_housing()
X=hausing.data
y=hausing.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

poly_feast=PolynomialFeatures(degree=2)
X_train_poly=poly_feast.fit_transform(X_train)
X_test_poly=poly_feast.transform(X_test)#eğitildi önceden ve ona göre dönüşümler yapar

poly_reg=LinearRegression()
poly_reg.fit(X_train_poly,y_train)
y_pred=poly_reg.predict(X_test_poly)
rmse=mean_squared_error(y_test, y_pred,squared=False)
print("rmse=",rmse)

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
rmse=mean_squared_error(y_test, y_pred,squared=False)
print("rmse=",rmse)








