# 📊 DTW-Based Clustering Pipeline for Production Bottlenecks

This project implements a robust time-series clustering pipeline for analyzing **bottleneck durations** in a manufacturing or production line environment. Leveraging **Dynamic Time Warping (DTW)**, the pipeline captures temporal similarities between operational patterns — enabling insightful grouping of production behaviors for diagnostics, optimization, or alert systems.

---

## 🔍 Problem Statement

In many production environments, bottlenecks (periods of delay or stoppage) are common but vary in pattern and duration. Traditional clustering (e.g., using average durations) ignores the **sequential and time-varying nature** of these events.

This project tackles the problem by:

- Treating bottleneck durations as **time series**
- Measuring similarity using **DTW**, which handles sequences of varying length and speed
- Applying clustering to discover **common patterns or anomalies** in stoppage behavior

---

## 🧠 Key Features

✅ Time-series segmentation per `(Line, Stoppage Reason, Shift Id)`  
✅ Outlier filtering using IQR  
✅ Dynamic Time Warping (DTW) distance computation  
✅ Hierarchical and density-based clustering (AHC, DBSCAN/HDBSCAN)  
✅ Cluster evaluation using silhouette score  
✅ Visual summaries: dendrograms, silhouette plots, cluster samples  
✅ Exportable cluster assignments and metrics  

---

## 📁 Project Structure

| File                             | Description                                              |
|----------------------------------|----------------------------------------------------------|
| `Stage1_Clustering_Pipeline copy.ipynb` | Main notebook with full preprocessing-to-export pipeline |
| `clustering_metrics_stage1.csv` | Summary of clustering evaluation metrics                 |
| `cluster_assignments_stage1.csv` | Cluster labels per time series group                     |
| `README.md`                     | Project documentation (this file)                        |
| `requirements.txt`             | Python package dependencies                              |

---

## 🧰 Technologies Used

- Python 3.8+ or 3.10 (Python 3.13 support still evolving for some packages)
- [tslearn](https://tslearn.readthedocs.io/en/stable/) – Time-series learning
- [scikit-learn](https://scikit-learn.org/) – Clustering algorithms and metrics
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) – Plotting
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) – Density-based clustering (optional)

---
