# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:20:20 2025
@author: Huzur Bilgisayar
"""
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np

# Rastgele 100 veri noktası oluştur
X = np.random.rand(100, 2)  # İki boyutlu giriş verisi
coef = np.array([3, 5])  # Doğrusal denklem katsayıları
y = np.random.rand(100) + np.dot(X, coef)  # Gürültü eklenmiş hedef değişken

# Modeli oluştur ve eğit
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 3D çizim için figür ve eksen oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Gerçek veri noktalarını çiz
ax.scatter(X[:, 0], X[:, 1], y, color='r', label="Gerçek Veriler")

# Eksenleri etiketle
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

# Tahmin yüzeyini oluşturmak için grid noktaları oluştur
x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
x1_flat = x1.flatten()
x2_flat = x2.flatten()
X_pred = np.column_stack((x1_flat, x2_flat))  # (100,2) boyutunda tahmin girdisi

# Modelden tahmin al
y_pred = lin_reg.predict(X_pred).reshape(x1.shape)

# Regresyon düzlemini çiz
ax.plot_surface(x1, x2, y_pred, alpha=0.5, color='cyan')

print("Katsayılar=",lin_reg.coef_)
print("kesişim=",lin_reg.intercept_)
# Başlık ekle
plt.title("Multi Linear Regression")
plt.legend()
plt.show()
