# Stage 1: Bottleneck Identification Pipeline
# ----------------------------------------------
# This script provides the skeleton for: 
#  - Loading and cleaning your Maggi production dataset
#  - Classifying active vs. non-active events
#  - Generating summary statistics per group
#  - Computing a distance matrix (Euclidean proxy for DTW)
#  - Clustering with Agglomerative Hierarchical Clustering (AHC)
#  - Clustering with density-based (DBSCAN) as an HDBSCAN proxy
#  - Evaluating and comparing cluster performance

import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
import matplotlib.pyplot as plt

# 1. Load data
# Replace path with your uploaded file

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(r"C:\Users\pbrin\Downloads\DATA WIP SORTED BASED ON ACTIVE STATE.xlsx", parse_dates=['Start Datetime', 'End Datetime'])
    return df

# 2. Clean & classify

def clean_and_classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Start Datetime', 'End Datetime'])
    df['is_active'] = np.where(df['Stoppage Category'] == 'Not Occupied', 0, 1)
    return df

# 3. Generate summary statistics per (Line, Stoppage Reason, Shift Id)

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(['Line', 'Stoppage Reason', 'Shift Id'])
    stats = pd.DataFrame({
        'mean': grouped['Bottleneck Duration Seconds'].mean(),
        'std': grouped['Bottleneck Duration Seconds'].std(),
        'max': grouped['Bottleneck Duration Seconds'].max(),
        'min': grouped['Bottleneck Duration Seconds'].min(),
        'count': grouped['Bottleneck Duration Seconds'].count()
    }).dropna()
    return stats

# 4. Prepare distance matrix

def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    D = pairwise_distances(X_scaled, metric='euclidean')
    return X_scaled, D

# 5. Agglomerative Hierarchical Clustering

def run_ahc(D: np.ndarray, n_clusters: int = 4):
    Z = linkage(D, method='complete')
    ahc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='complete'
    )
    labels = ahc.fit_predict(D)
    return Z, labels

# 6. Density-based clustering via DBSCAN (HDBSCAN proxy)

def run_dbscan(D: np.ndarray, eps: float, min_samples: int = 5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(D)
    return labels

# 7. Evaluate clustering

def evaluate_clustering(D: np.ndarray, X_feat: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    result['Silhouette'] = silhouette_score(D, labels, metric='precomputed') if len(set(labels)) > 1 else np.nan
    mask = labels >= 0
    if len(set(labels[mask])) > 1:
        result['Davies-Bouldin'] = davies_bouldin_score(X_feat[mask], labels[mask])
        result['Calinski-Harabasz'] = calinski_harabasz_score(X_feat[mask], labels[mask])
    else:
        result['Davies-Bouldin'] = np.nan
        result['Calinski-Harabasz'] = np.nan
    return result

# 8. Visualization helpers

def plot_dendrogram(Z):
    plt.figure(figsize=(10, 5))
    dendrogram(Z, color_threshold=0)
    plt.title('Dendrogram: AHC (Complete Linkage)')
    plt.xlabel('Group index')
    plt.ylabel('Distance')
    plt.show()

# 9. Main flow
if __name__ == '__main__':
    path = 'DATA WIP SORTED BASED ON ACTIVE STATE.xlsx'  # update accordingly
    df = load_dataset(path)
    df_clean = clean_and_classify(df)
    stats = generate_summary(df_clean)
    X_feat, D = compute_distance_matrix(stats.values)

    # AHC
    Z, labels_ahc = run_ahc(D)

    # DBSCAN
    eps = np.median(D)
    labels_db = run_dbscan(D, eps=eps)

    # Evaluate
    metrics = {
        'AHC': evaluate_clustering(D, X_feat, labels_ahc),
        'DBSCAN': evaluate_clustering(D, X_feat, labels_db)
    }
    metrics_df = pd.DataFrame(metrics).T

    # Display metrics via markdown print
    print('Clustering Performance Metrics:')
    print(metrics_df.to_markdown())

    # Dendrogram
    plot_dendrogram(Z)

    # Cluster counts
    print('AHC cluster counts:', np.bincount(labels_ahc))
    print('DBSCAN cluster counts (noise = -1):', np.bincount(labels_db + 1))
