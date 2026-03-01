from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_blobs, y_true = make_blobs(n_samples=600, centers=4,
                             cluster_std=0.5, random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true)
plt.title('Prawdziwe etykiety')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.show()