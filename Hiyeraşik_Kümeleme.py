# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:17:54 2025
@author: Huzur Bilgisayar
"""


from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt 

# Örnek veri oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Veriyi çizdirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Örnek Veri")

# Bağlantı yöntemleri
linkage_methods = ["ward", "single", "average", "complete"]

plt.figure(figsize=(12, 6))

for i, method in enumerate(linkage_methods, 1):
    model = AgglomerativeClustering(n_clusters=4, linkage=method)
    cluster_labels = model.fit_predict(X)

    # Dendrogram
    plt.subplot(2, 4, i)
    plt.title(f"{method.capitalize()} Linkage Dendrogram")
    dendrogram(linkage(X, method=method), no_labels=True)
    plt.xlabel("Veri Noktaları")
    plt.ylabel("Uzaklık")

    # Kümeleme grafiği
    plt.subplot(2, 4, i + 4)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis")
    plt.title(f"{method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")

plt.tight_layout()
plt.show()
