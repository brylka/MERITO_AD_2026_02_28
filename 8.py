from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Dane z klastrami o różnej gęstości
X_varied, y_varied = make_blobs(n_samples=[100, 50, 150],
                                 centers=[[0, 0], [3, 3], [6, 1]],
                                 cluster_std=[0.5, 1.5, 0.3],
                                 random_state=42)

# DBSCAN na danych o różnej gęstości
dbscan_varied = DBSCAN(eps=0.5, min_samples=5)
y_dbscan_varied = dbscan_varied.fit_predict(X_varied)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied, cmap='viridis', alpha=0.6)
plt.title('Prawdziwe klastry (różna gęstość)')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')

plt.subplot(1, 2, 2)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_dbscan_varied, cmap='viridis', alpha=0.6)
outliers_varied = X_varied[y_dbscan_varied == -1]
if len(outliers_varied) > 0:
    plt.scatter(outliers_varied[:, 0], outliers_varied[:, 1], c='red',
                marker='x', s=100, linewidths=2, label='Outliers')
plt.title('DBSCAN (problemy z różną gęstością)')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.legend()

plt.tight_layout()
plt.show()
