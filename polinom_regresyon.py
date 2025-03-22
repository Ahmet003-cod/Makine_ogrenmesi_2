# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:32:07 2025

@author: Huzur Bilgisayar
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#veri oluşturma
X=6*np.random.rand(100,1)
y=15+3*X**2+np.random.rand(100,1)#  3x^2+15 polinomu oluştu

#plt.scatter(X, y)
poly_feat=PolynomialFeatures(degree=2)#derecesi belirlendi
X_poly=poly_feat.fit_transform(X)
#eğitim modeli
poly_reg=LinearRegression()
X_ploy_reg=poly_reg.fit(X_poly,y)
#ilk eğitim modelinin grafiği
plt.scatter(X, y,color="blue")
#Test için veri oluşumu
X_test=np.linspace(0,6,100).reshape(-1,1)
X_test_poly=poly_feat.transform(X_test)
y_pred=poly_reg.predict(X_test_poly)
#Grafik oluşumu polinom
plt.plot(X_test,y_pred,color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polinom Regresyon modeli")
plt.show()

