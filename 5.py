from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = load_iris()
X, y_true = data.data[:, :2], data.target

kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

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
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', c='red', s=200, label='Centroidy')
plt.title('Klasteryzacja K-means')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.legend()
plt.show()