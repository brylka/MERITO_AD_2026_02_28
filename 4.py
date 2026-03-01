from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import  silhouette_score
import matplotlib.pyplot as plt

X_blobs, y_true = make_blobs(n_samples=300, centers=8,
                             cluster_std=0.5, random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true)
plt.title('Prawdziwe etykiety')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.show()

inertias = []
silhouette_scores = []
K_range = range(2,11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_blobs)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, kmeans.labels_))

print(inertias)


plt.figure(figsize=(8,5))
plt.plot(K_range, silhouette_scores)
plt.title('Współczynnik sylwetkowy dla liczby klastrów (k)')
plt.xlabel('Liczba klastrów')
plt.ylabel('Wartość współczynnika')
plt.grid(True)
plt.show()

