# Streamlit App: Live Monitoring Dashboard for DTW-Based Clustering

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import plotly.express as px

# --- Load Data ---
st.set_page_config(layout="wide")
st.title("🔍 Production Bottleneck Clustering Dashboard")

@st.cache_data

def load_data():
    cluster_df = pd.read_csv("cluster_assignments_stage1.csv", index_col=[0, 1, 2])
    metrics_df = pd.read_csv("clustering_metrics_stage1.csv", index_col=0)
    return cluster_df, metrics_df

cluster_df, metrics_df = load_data()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔧 Filter View")
    lines = st.multiselect("Select Lines", options=cluster_df.index.get_level_values("Line").unique(), default=None)
    reasons = st.multiselect("Select Stoppage Reasons", options=cluster_df.index.get_level_values("Stoppage Reason").unique(), default=None)
    shifts = st.multiselect("Select Shifts", options=cluster_df.index.get_level_values("Shift Id").unique(), default=None)

    st.divider()
    show_time_series = st.checkbox("Show Time Series Samples", value=True)
    show_dendro = st.checkbox("Show Dendrogram", value=False)

# --- Filtered Data ---
filtered = cluster_df.copy()
if lines:
    filtered = filtered[filtered.index.get_level_values("Line").isin(lines)]
if reasons:
    filtered = filtered[filtered.index.get_level_values("Stoppage Reason").isin(reasons)]
if shifts:
    filtered = filtered[filtered.index.get_level_values("Shift Id").isin(shifts)]

st.subheader("📊 Cluster Assignments")
st.dataframe(filtered.reset_index(), use_container_width=True)

# --- Metrics ---
st.subheader("📈 Clustering Evaluation Metrics")
st.dataframe(metrics_df.style.format("{:.3f}"))

# --- Cluster Distribution ---
st.subheader("📌 Cluster Distribution")
fig = px.histogram(filtered.reset_index(), x="AHC_label", color="AHC_label", nbins=20, title="Cluster Count by AHC")
st.plotly_chart(fig, use_container_width=True)

# --- Optional Time Series Sample Display ---
if show_time_series:
    st.subheader("📉 Sample Time Series per Cluster")
    ts_dict = np.load("ts_dict_stage1.npy", allow_pickle=True).item()  # Pre-saved dictionary of time series

    for cluster_id in sorted(filtered["AHC_label"].unique())[:3]:
        st.markdown(f"### Cluster {cluster_id}")
        keys = [k for k in ts_dict.keys() if k in filtered.index and filtered.loc[k, "AHC_label"] == cluster_id]
        fig, ax = plt.subplots(figsize=(8, 3))
        for key in keys[:5]:
            ax.plot(ts_dict[key], alpha=0.7)
        ax.set_xlabel("Event Index")
        ax.set_ylabel("Duration (s)")
        st.pyplot(fig)

# --- Optional Dendrogram ---
if show_dendro:
    st.subheader("🌳 Dendrogram (Static Snapshot)")
    st.image("dendrogram_snapshot.png", caption="Dendrogram from DTW Distance Matrix")

st.success("Dashboard loaded successfully.")
