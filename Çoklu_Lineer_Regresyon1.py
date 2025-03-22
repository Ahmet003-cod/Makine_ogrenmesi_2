# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:09:42 2025

@author: Huzur Bilgisayar
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

diyabet=load_diabetes()
X=diyabet.data
y=diyabet.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)

rmse=mean_squared_error(y_test, y_pred,squared=False)#ardaki farkın kareiolur
print("rmse=",rmse)
import matplotlib.pyplot as plt

# Gerçek değerler ve tahminler
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)  # Gerçek vs Tahmin edilen değerler
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # y=x çizgisi
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler (Lineer Regresyon)')
plt.grid(True)
plt.show()

