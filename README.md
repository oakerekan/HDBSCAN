# Bottleneck Analysis & Forecasting

This repository provides a comprehensive Jupyter notebook pipeline for identifying and forecasting production bottlenecks in a Maggi production plant. It is organized into two stages:

- **Stage 1**: Unsupervised clustering of bottleneck events using Agglomerative Hierarchical Clustering (AHC) and density-based methods, with extensive validation and visualization.  
- **Stage 2**: Supervised forecasting of future bottleneck durations using Recurrent Neural Networks (LSTM & GRU).

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Notebook Structure](#notebook-structure)  
3. [Requirements](#requirements)  
4. [Setup Instructions](#setup-instructions)  
5. [Usage](#usage)  
6. [Output Files](#output-files)  
7. [Next Steps](#next-steps)

---

## Project Overview

This analysis tackles two key challenges:

1. **Bottleneck Identification**  
   Extract meaningful patterns in machine stoppage data to group similar behaviors (e.g., long active periods, frequent idle cycles) that represent production constraints.

2. **Bottleneck Forecasting**  
   Build predictive models to forecast the next bottleneck duration, enabling proactive maintenance and scheduling.

Data is sourced from stoppage logs with fields such as:
`Line`, `Stoppage Category`, `Stoppage Reason`, `Start Datetime`, `End Datetime`, and `Bottleneck Duration Seconds`.

---

## Notebook Structure

The main notebook (`Stage1_Clustering_Pipeline.ipynb`) is organized as follows:

1. **Imports**  
   Load Python libraries (pandas, numpy, scikit-learn, seaborn, tslearn, tf-keras).

2. **Data Loading & Cleaning**  
   Read the Excel file, validate columns, drop missing values, classify active vs. non-active events.

3. **Advanced Preprocessing**  
   IQR-based outlier removal and optional smoothing of duration signals.

4. **Summary Statistics & Time-Series Extraction**  
   Create `stats` DataFrame (mean, std, min, max, count) and raw time-series dict for DTW.

5. **Distance Matrices**  
   Compute Euclidean distances on `stats` and optional DTW distances on raw series.  
   - **Heatmap** of distance matrix  
   - **Elbow plot** (KMeans WCSS)

6. **Parameter Tuning for AHC**  
   Use silhouette analysis to pick the optimal number of clusters.

7. **Clustering Execution**  
   Perform AHC and density-based clustering (HDBSCAN/DBSCAN fallback).

8. **Evaluation Metrics**  
   Compute silhouette, Davies–Bouldin, and Calinski–Harabasz indices for both methods.

9. **Visualizations**  
   Dendrogram, silhouette curve, and time-series overlays for cluster inspection.

10. **Export Results**  
    Save clustering metrics and assignments to CSV files.

11. **Additional Analyses**  
    Suggestions and code stubs for advanced PhD-level extensions (t-SNE, UMAP, bootstrap stability, survival analysis, etc.).

12. **Stage 2: Predictive Modeling**  
    Data preparation, LSTM & GRU model training, evaluation, and comparison.

---

## Requirements

- **Python 3.8+**  
- **Jupyter Notebook** or **JupyterLab**  
- Required packages (install via `pip`):  
  ```bash
  pip install pandas numpy matplotlib seaborn scipy scikit-learn tensorflow tslearn[hdf5] hdbscan umap-learn ruptures lifelines shap
