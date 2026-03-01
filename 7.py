from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import  silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#X_blobs, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=1.5, random_state=42)
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)
#X, y_true = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# kmeans = KMeans(n_clusters=2, random_state=42)
# y_kmeans = kmeans.fit_predict(X_blobs)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(16, 5))
plt.subplot(1,2,1)
plt.scatter(X[:, 0], X[:, 1], c=y_true)
plt.title('Prawdziwe etykiety')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
# plt.show()
#
# plt.figure(figsize=(8, 5))
plt.subplot(1,2,2)
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#             marker='X', c='red', s=200, label='Centroidy')
plt.title('Klasteryzacja DBSCAN')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.legend()
plt.show()