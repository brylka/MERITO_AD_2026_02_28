from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==============================================================================
# 1. WCZYTANIE DANYCH IRIS
# ==============================================================================
iris = load_iris()
X = iris.data  # 4 cechy: długość/szerokość działki i płatka
y_true = iris.target  # 3 gatunki (prawdziwe etykiety)
feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset Iris:")
print(f"- Liczba próbek: {X.shape[0]}")
print(f"- Liczba cech: {X.shape[1]}")
print(f"- Cechy: {feature_names}")
print(f"- Gatunki: {target_names}")
print(f"- Prawdziwa liczba klastrów: 3\n")

# ==============================================================================
# 2. TESTOWANIE K-MEANS DLA RÓŻNYCH WARTOŚCI k
# ==============================================================================
print("Testuję K-means dla k = 2, 3, 4, 5...")

K_range = range(2, 6)  # k = 2, 3, 4, 5
inertias = []
silhouette_scores = []
davies_bouldin_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X, kmeans.labels_))

# ==============================================================================
# 3. WIZUALIZACJA: METODA ŁOKCIA I WSPÓŁCZYNNIK SYLWETKOWY
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Wykres 1: Metoda łokcia (Inercja)
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=10)
axes[0].set_xlabel('Liczba klastrów (k)', fontsize=12)
axes[0].set_ylabel('Inercja', fontsize=12)
axes[0].set_title('Metoda Łokcia\n(szukamy "łokcia" - punktu przegięcia)', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(K_range)
# Zaznaczenie k=3
axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='k=3 (prawdziwe)')
axes[0].legend()

# Wykres 2: Współczynnik sylwetkowy
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=10)
axes[1].set_xlabel('Liczba klastrów (k)', fontsize=12)
axes[1].set_ylabel('Współczynnik sylwetkowy', fontsize=12)
axes[1].set_title('Współczynnik Sylwetkowy\n(wyższy = lepiej)', fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(K_range)
axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='k=3 (prawdziwe)')
axes[1].legend()
# Zaznaczenie najlepszego wyniku
best_k_silhouette = K_range[np.argmax(silhouette_scores)]
axes[1].scatter([best_k_silhouette], [max(silhouette_scores)],
                color='gold', s=200, zorder=5, edgecolor='orange', linewidth=2)

# Wykres 3: Davies-Bouldin Index
axes[2].plot(K_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=10)
axes[2].set_xlabel('Liczba klastrów (k)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[2].set_title('Davies-Bouldin Index\n(niższy = lepiej)', fontsize=13)
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(K_range)
axes[2].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='k=3 (prawdziwe)')
axes[2].legend()
# Zaznaczenie najlepszego wyniku
best_k_db = K_range[np.argmin(davies_bouldin_scores)]
axes[2].scatter([best_k_db], [min(davies_bouldin_scores)],
                color='gold', s=200, zorder=5, edgecolor='orange', linewidth=2)

plt.tight_layout()
plt.show()

# ==============================================================================
# 4. ANALIZA WYNIKÓW
# ==============================================================================
print("\n" + "=" * 70)
print("WYNIKI DLA RÓŻNYCH WARTOŚCI k:")
print("=" * 70)
print(f"{'k':<5} {'Inercja':<15} {'Silhouette':<15} {'Davies-Bouldin':<15}")
print("-" * 70)
for i, k in enumerate(K_range):
    print(f"{k:<5} {inertias[i]:<15.2f} {silhouette_scores[i]:<15.4f} {davies_bouldin_scores[i]:<15.4f}")

print("\n" + "=" * 70)
print("KTÓRE k JEST OPTYMALNE?")
print("=" * 70)
print(f"Według Silhouette Score: k = {best_k_silhouette} (najwyższy wynik)")
print(f"Według Davies-Bouldin: k = {best_k_db} (najniższy wynik)")
print(f"Prawdziwa liczba gatunków: k = 3")
print("\nWNIOSKI:")
print("- K=2 jest za mało (metrics pokazują, że można podzielić bardziej)")
print("- K=3 odpowiada prawdziwej liczbie gatunków - dobry kompromis")
print("- K=2 ma najwyższy Silhouette, ale to za mało klastrów dla Iris")
print("- Metoda łokcia sugeruje k=3 (punkt przegięcia)")
print("=" * 70 + "\n")

# ==============================================================================
# 5. K-MEANS Z OPTYMALNYM k=3
# ==============================================================================
optimal_k = 3
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_kmeans = kmeans_optimal.fit_predict(X)

# ==============================================================================
# 6. WIZUALIZACJA 4D DANYCH - METODA 1: PCA (redukcja do 2D)
# ==============================================================================
print("Redukuję 4 wymiary do 2D za pomocą PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prawdziwe etykiety
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true,
                           cmap='viridis', alpha=0.7, s=100, edgecolor='black')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)', fontsize=12)
axes[0].set_title('Prawdziwe gatunki (PCA 2D)', fontsize=13)
axes[0].grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=axes[0], ticks=[0, 1, 2])
cbar1.set_ticklabels(target_names)

# K-means (k=3)
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans,
                           cmap='viridis', alpha=0.7, s=100, edgecolor='black')
# Dodanie centroidów
centroids_pca = pca.transform(kmeans_optimal.cluster_centers_)
axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                s=500, c='red', marker='X', edgecolors='black',
                linewidths=3, label='Centroidy', zorder=5)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)', fontsize=12)
axes[1].set_title(f'K-means (k={optimal_k}) - PCA 2D', fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].legend()
cbar2 = plt.colorbar(scatter2, ax=axes[1], ticks=[0, 1, 2])
cbar2.set_label('Klaster', fontsize=11)

plt.tight_layout()
plt.show()

print(f"PCA: Zachowano {pca.explained_variance_ratio_.sum():.1%} całkowitej wariancji")

# ==============================================================================
# 7. WIZUALIZACJA 4D - METODA 2: MACIERZ SCATTER PLOTÓW (wszystkie pary cech)
# ==============================================================================
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i in range(4):
    for j in range(4):
        ax = axes[i, j]

        if i == j:
            # Histogram na przekątnej
            ax.hist(X[:, i], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_ylabel('Częstość', fontsize=9)
        else:
            # Scatter plot z kolorami według K-means
            scatter = ax.scatter(X[:, j], X[:, i], c=y_kmeans,
                                 cmap='viridis', alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
            # Dodanie centroidów
            ax.scatter(kmeans_optimal.cluster_centers_[:, j],
                       kmeans_optimal.cluster_centers_[:, i],
                       s=200, c='red', marker='X', edgecolors='black', linewidths=2)

        # Etykiety
        if i == 3:
            ax.set_xlabel(feature_names[j], fontsize=10)
        else:
            ax.set_xticklabels([])

        if j == 0:
            ax.set_ylabel(feature_names[i], fontsize=10)
        else:
            ax.set_yticklabels([])

        ax.grid(True, alpha=0.2)

plt.suptitle(f'Macierz Scatter Plotów - K-means (k={optimal_k})\nCzerwone X = Centroidy',
             fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

# ==============================================================================
# 8. WIZUALIZACJA 4D - METODA 3: SCATTER 3D Z KOLOREM jako 4. wymiar
# ==============================================================================
fig = plt.figure(figsize=(16, 6))

# Wykres 1: Prawdziwe gatunki
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2],
                       c=X[:, 3], cmap='coolwarm',
                       s=100, alpha=0.7, edgecolor='black')
ax1.set_xlabel(feature_names[0], fontsize=10)
ax1.set_ylabel(feature_names[1], fontsize=10)
ax1.set_zlabel(feature_names[2], fontsize=10)
ax1.set_title('Iris 4D: 3 osie + kolor = 4. cecha\n(Prawdziwe dane)', fontsize=12)
cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.8)
cbar1.set_label(feature_names[3], fontsize=10)

# Wykres 2: K-means
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2],
                       c=y_kmeans, cmap='viridis',
                       s=100, alpha=0.7, edgecolor='black')
# Centroidy
ax2.scatter(kmeans_optimal.cluster_centers_[:, 0],
            kmeans_optimal.cluster_centers_[:, 1],
            kmeans_optimal.cluster_centers_[:, 2],
            s=500, c='red', marker='X', edgecolors='black',
            linewidths=3, label='Centroidy')
ax2.set_xlabel(feature_names[0], fontsize=10)
ax2.set_ylabel(feature_names[1], fontsize=10)
ax2.set_zlabel(feature_names[2], fontsize=10)
ax2.set_title(f'K-means (k={optimal_k})', fontsize=12)
ax2.legend()
cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8, ticks=[0, 1, 2])
cbar2.set_label('Klaster', fontsize=10)

plt.tight_layout()
plt.show()

# ==============================================================================
# 9. PORÓWNANIE Z PRAWDZIWYMI ETYKIETAMI
# ==============================================================================
from sklearn.metrics import confusion_matrix, adjusted_rand_score

print("\n" + "=" * 70)
print("PORÓWNANIE K-MEANS (k=3) Z PRAWDZIWYMI GATUNKAMI:")
print("=" * 70)

# Adjusted Rand Index - mierzy zgodność klastrów z prawdziwymi etykietami
ari = adjusted_rand_score(y_true, y_kmeans)
print(f"Adjusted Rand Index: {ari:.4f}")
print("(1.0 = idealne dopasowanie, 0.0 = losowe)")

# Macierz pomyłek
conf_matrix = confusion_matrix(y_true, y_kmeans)
print("\nMacierz pomyłek (prawdziwe vs K-means):")
print(conf_matrix)
print("\nInterpretacja:")
print("- Wiersze = prawdziwe gatunki")
print("- Kolumny = klastry K-means")
print("- K-means może nadać inne numery klastrów niż prawdziwe etykiety!")

# ==============================================================================
# 10. PODSUMOWANIE
# ==============================================================================
print("\n" + "=" * 70)
print("PODSUMOWANIE:")
print("=" * 70)
print(f"Optymalne k = {optimal_k} (zgodne z prawdziwą liczbą gatunków)")
print(f"Silhouette Score: {silhouette_scores[optimal_k - 2]:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_scores[optimal_k - 2]:.4f}")
print(f"Zgodność z prawdziwymi etykietami (ARI): {ari:.4f}")
print("\nDLACZEGO k=3 jest optymalne?")
print("1. Metoda łokcia pokazuje punkt przegięcia przy k=3")
print("2. Współczynnik sylwetkowy jest wysoki")
print("3. Odpowiada prawdziwej liczbie gatunków")
print("4. K=2 byłoby zbyt uproszczone, k=4-5 to overfitting")
print("=" * 70)
