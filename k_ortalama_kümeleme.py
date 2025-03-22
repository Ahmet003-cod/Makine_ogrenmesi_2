# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:33:07 2025

@author: Huzur Bilgisayar
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

X,_=make_blobs(n_samples=300,centers=5,cluster_std=0.9,random_state=42)#iki değişkene atamak zorunda olduğumuz için _ boş kullandık

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("Örnek Veri")
kmeans=KMeans(n_clusters=5)
kmeans.fit(X)
labels=kmeans.labels_
plt.figure()
plt.scatter(X[:,0],X[:,1],c=labels,cmap="viridis")
center=kmeans.cluster_centers_
plt.scatter(center[:,0],center[:,1],c="red",marker="X")
plt.title("X_Means")