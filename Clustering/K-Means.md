# 📊 K-Means Clustering — Complete Theory & Evaluation Guide

> **Unsupervised Machine Learning Series** | Author: **Ashok**

---

## 📌 Table of Contents

- [What is Clustering?](#-1-what-is-clustering)
- [What is K-Means?](#-2-what-is-k-means)
- [Algorithm Steps](#️-3-algorithm-steps)
- [Evaluation Metrics](#-4-evaluation-metrics)
- [Choosing Optimal K](#-5-choosing-optimal-k)
- [Python Implementation](#-6-python-implementation)
- [Assumptions & Limitations](#️-7-assumptions--limitations)
- [Quick Reference](#-8-quick-reference-summary)
- [How to Run](#-9-how-to-run)

---

## 🧠 1. What is Clustering?

Clustering is an **unsupervised learning technique** that groups similar data points together **without labeled outputs**. The algorithm discovers natural structure in data purely from feature similarity.

### Real-World Applications

| Domain | Use Case |
|---|---|
| Marketing | Customer segmentation by purchase behaviour |
| Computer Vision | Image compression via pixel clustering |
| NLP | Document grouping by topic |
| Genomics | Gene expression profiling |
| Security | Anomaly / intrusion detection |

---

## 🎯 2. What is K-Means?

K-Means is a **centroid-based clustering algorithm** that partitions data into **K non-overlapping clusters** by minimising within-cluster variance. Each cluster is represented by its centroid — the arithmetic mean of all assigned points.

### 🔢 Objective Function (WCSS)

K-Means minimises the **Within-Cluster Sum of Squares**:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

| Symbol | Meaning |
|---|---|
| $K$ | Total number of clusters |
| $\mu_k$ | Centroid (mean) of cluster $k$ |
| $x_i$ | Data point assigned to cluster $k$ |
| $\|x_i - \mu_k\|^2$ | Squared Euclidean distance |

> Also known as: **Inertia** or **WCSS**

---

### ⚡ K-Means++ Initialisation

Standard K-Means uses random initialisation, which can converge to poor local minima. **K-Means++** improves this:

1. Choose first centroid **uniformly at random** from the data
2. For each remaining point, compute its distance to the **nearest chosen centroid**
3. Choose the next centroid with probability **proportional to distance²**
4. Repeat until K centroids are placed, then run standard K-Means

> ✅ Sklearn uses K-Means++ by default: `KMeans(init="k-means++")`  
> ✅ Typically **2–5× faster** convergence with better final WCSS

---

## ⚙️ 3. Algorithm Steps

```
┌─────────────────────────────────────┐
│         START — Choose K            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 1: Initialize K Centroids     │
│  (randomly or with K-Means++)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 2: Assign Each Point          │
│  to Nearest Centroid                │
│  (via Euclidean distance)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 3: Recalculate Centroids      │
│  μₖ = mean of all points in Cₖ     │
└──────────────┬──────────────────────┘
               │
               ▼
       ╔═══════════════╗
       ║  Converged?   ║ ──── NO ────┐
       ║ (stable?)     ║             │
       ╚═══════╤═══════╝             │
               │ YES                 │
               ▼                     │
┌─────────────────────────────────────┐ │
│  Final Cluster Assignments          │ │
│  → Evaluate with metrics            │ │
└─────────────────────────────────────┘ │
                                        │
         ◄──────── Repeat ─────────────┘
```

### Complexity Analysis

| | Complexity |
|---|---|
| **Time** | O(n · K · d · I) |
| **Space** | O(n · d + K · d) |

Where `n` = data points, `K` = clusters, `d` = features, `I` = iterations

---

## 📊 4. Evaluation Metrics

### Overview Table

| Metric | Formula / Basis | Range | Optimum | External? |
|---|---|---|---|---|
| **Inertia (WCSS)** | Σ‖x − μ‖² | 0 → ∞ | Lower ↓ | No |
| **Silhouette Score** | (b − a) / max(a, b) | −1 to +1 | Higher ↑ | No |
| **Calinski-Harabasz** | Between / within variance ratio | 0 → ∞ | Higher ↑ | No |
| **Davies-Bouldin** | Avg cluster similarity ratio | 0 → ∞ | Lower ↓ | No |
| **Adjusted Rand Index** | Comparison with true labels | −1 to +1 | Higher ↑ | ✅ Yes |

---

### 1️⃣ Inertia (WCSS)

- Measures **cluster compactness** — how tightly points cluster around their centroid
- Always decreases as K increases (not reliable alone for choosing K)
- Used in the **Elbow Method**

---

### 2️⃣ Silhouette Score

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\ b(i))}$$

| Variable | Meaning |
|---|---|
| `a(i)` | Avg distance from point `i` to all other points **in the same cluster** |
| `b(i)` | Avg distance from point `i` to all points in the **nearest other cluster** |

| Score | Interpretation |
|---|---|
| ≈ +1 | Well-clustered |
| ≈  0 | Near cluster boundary |
| ≈ −1 | Likely misclassified |

---

### 3️⃣ Calinski-Harabasz Index

$$CH = \frac{B_k / (K-1)}{W_k / (n-K)}$$

- **Between-cluster variance** / **Within-cluster variance**
- Higher score = more dense, well-separated clusters

---

### 4️⃣ Davies-Bouldin Index

$$DB = \frac{1}{K} \sum_{k=1}^{K} \max_{j \neq k} \left[ \frac{\sigma_k + \sigma_j}{d(\mu_k, \mu_j)} \right]$$

| Variable | Meaning |
|---|---|
| `σₖ` | Avg distance of points in cluster `k` from its centroid |
| `d(μₖ, μⱼ)` | Distance between centroids `k` and `j` |

- Lower DB = tighter, more separated clusters ✅

---

### 5️⃣ Adjusted Rand Index (External)

- Compares predicted cluster labels to **ground truth labels**
- Corrected for chance — ARI = 1.0 is a perfect match
- Requires labeled data (not always available)

---

## 📈 5. Choosing Optimal K

### 🔹 Elbow Method

Plot **K vs Inertia** and find the "elbow" — the point where adding more clusters yields diminishing returns.

```
Inertia
  │
  │ ●
  │   ●
  │      ●
  │          ●  ← Elbow
  │              ● ● ● ●
  └─────────────────────── K
       1  2  3  4  5  6
```

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(K_range, inertia, 'bo-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.show()
```

---

### 🔹 Silhouette Method

Compute the **mean silhouette score** for each K. Pick the K with the **highest score**.

```python
from sklearn.metrics import silhouette_score

sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

best_k = range(2, 11)[sil_scores.index(max(sil_scores))]
print(f"Optimal K (Silhouette): {best_k}")
```

### Comparison

| Method | Basis | Strength | Limitation |
|---|---|---|---|
| **Elbow** | WCSS vs K plot | Fast, intuitive | Elbow can be ambiguous |
| **Silhouette** | Cluster cohesion/separation | Reliable validation | O(n²) — slow on large data |

---

## 💻 6. Python Implementation

### Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy
scikit-learn
matplotlib
```

---

### Full Implementation (`main.py`)

```python
# ─────────────────────────────────────────────────────────────
#  K-Means Clustering — Full Implementation
#  Author: Ashok
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# 1. Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=4,
                        cluster_std=0.8, random_state=42)

# 2. Scale features (critical for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find optimal K via Elbow + Silhouette
inertia, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# 4. Fit final model with best K
best_k = list(K_range)[sil_scores.index(max(sil_scores))]
km_final = KMeans(n_clusters=best_k, init="k-means++",
                   n_init=10, random_state=42)
labels = km_final.fit_predict(X_scaled)

# 5. Evaluation metrics
print(f"{'─'*45}")
print(f"  Optimal K (Silhouette)  : {best_k}")
print(f"  Inertia (WCSS)          : {km_final.inertia_:.2f}")
print(f"  Silhouette Score        : {silhouette_score(X_scaled, labels):.4f}")
print(f"  Calinski-Harabasz       : {calinski_harabasz_score(X_scaled, labels):.2f}")
print(f"  Davies-Bouldin          : {davies_bouldin_score(X_scaled, labels):.4f}")
print(f"{'─'*45}")

# 6. Visualise clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Elbow curve
axes[0].plot(K_range, inertia, 'bo-')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

# Silhouette curve
axes[1].plot(K_range, sil_scores, 'go-')
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

# Final clusters
axes[2].scatter(X_scaled[:,0], X_scaled[:,1],
                c=labels, cmap='viridis', alpha=0.7, s=30)
axes[2].scatter(km_final.cluster_centers_[:,0],
                km_final.cluster_centers_[:,1],
                c='red', marker='X', s=200, label='Centroids')
axes[2].set_title(f"K-Means Clusters (K={best_k})")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## ⚠️ 7. Assumptions & Limitations

### ✅ Key Assumptions

- Clusters are **spherical** and roughly **equal in size**
- **Euclidean distance** is an appropriate similarity measure
- **K is known** (or can be estimated) in advance
- Features are on the **same scale** — always standardise first

### ❌ Limitations

| Limitation | Details |
|---|---|
| Sensitive to initialisation | Different seeds → different results (mitigated by K-Means++) |
| Requires K upfront | No automatic cluster count discovery |
| Assumes convex clusters | Cannot find ring/crescent-shaped clusters |
| Outlier sensitivity | Outliers can significantly distort centroids |
| Unequal cluster sizes | May misassign points in imbalanced datasets |

### 🔄 Alternatives

| Algorithm | Best For |
|---|---|
| **DBSCAN** | Arbitrary shapes, noise/outlier handling |
| **GMM** | Soft assignments, ellipsoidal clusters |
| **Agglomerative** | No K required, hierarchical structure |
| **Spectral** | Non-convex, manifold-shaped clusters |

---

## 📝 8. Quick Reference Summary

| Topic | Key Points |
|---|---|
| **Algorithm Type** | Centroid-based, partitional, unsupervised |
| **Objective** | Minimise WCSS (Within-Cluster Sum of Squares) |
| **Initialisation** | Random or K-Means++ (sklearn default) |
| **Distance Metric** | Euclidean distance ‖x − μ‖ |
| **Convergence** | When centroids stop moving (or move < ε) |
| **Complexity** | O(n · K · d · I) |
| **Best K — Elbow** | Plot WCSS vs K; look for the "knee" |
| **Best K — Silhouette** | Maximise mean silhouette coefficient |
| **Internal Metrics** | WCSS, Silhouette, Calinski-Harabasz, Davies-Bouldin |
| **External Metric** | Adjusted Rand Index (requires ground truth) |
| **Key Assumption** | Spherical, similarly-sized clusters |
| **Preprocessing** | `StandardScaler` essential before running K-Means |

---



*Unsupervised Machine Learning Series*

</div>
