# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 19:28:41 2025
@author: Huzur Bilgisayar
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#veri oluştur
X =np.random.rand(100,1)
y=3 +4*X+0.5*np.random.rand(100,1)

lin_reg=LinearRegression()
lin_reg.fit(X, y)
plt.figure()
plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X),color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Lineer Regresyon")

a1=lin_reg.coef_[0][0]# denklemin  katsayısını verir  a0+a1x
a0=lin_reg.intercept_[0]#denlemin sabit sayisini verir
print("a1=",a1)

print("a0=",a0)

for i in range(100):
    y_=a0+a1*X
    plt.plot(X,y_,color="green",alpha=0.8)