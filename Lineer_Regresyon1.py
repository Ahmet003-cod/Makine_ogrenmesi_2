# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:06:57 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
#Veri seti alma
diyabet=load_diabetes()
#Eğitim için hazırlık
diyabet_X,diyabet_y=load_diabetes(return_X_y=True)

diyabet_X=diyabet_X[:,np.newaxis,3]#☻ sadece 3.sütün olan bmi alıyor
#eğitim bölümü
diyabet_X_train=diyabet_X[:-20]
#test bölümü
diyabet_X_test=diyabet_X[-20:]
#eğitim bölümü
diyabet_y_train=diyabet_y[:-20]
#test bölümü
diyabet_y_test=diyabet_y[-20:]
lin_reg=LinearRegression()
lin_reg.fit(diyabet_X_train,diyabet_y_train)
diyabet_y_pred=lin_reg.predict(diyabet_X_test)

mse=mean_squared_error(diyabet_y_test,diyabet_y_pred)
print("mse=",mse)

R2=r2_score(diyabet_y_test,diyabet_y_pred)
print("R2=",R2)

plt.scatter(diyabet_X_test,diyabet_y_test,color="blue")
plt.plot(diyabet_X_test,diyabet_y_pred,color="red")