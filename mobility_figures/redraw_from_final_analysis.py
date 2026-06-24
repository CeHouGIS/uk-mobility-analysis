#!/usr/bin/env python
"""
Redraw the original cluster_profiles.py figures from fig_output/final_analysis_data.csv,
USING THE EXISTING cluster_2020 / cluster_2021 labels in the file (no re-clustering),
so the cluster contents/numbering match the original fig_output reference figures.

Plotting style is copied verbatim from src/s4_clustering/cluster_profiles.py.
Ternary uses LAD-normalised days (the cluster_profiles.py representation).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

DATA = "/workplace/Mob/fig_output/final_analysis_data.csv"
OUT = "/workplace/Mob/mobility_figures/faithful_latest"
os.makedirs(OUT, exist_ok=True)
BASE = ["days_t1", "days_t3", "days_t5", "days_t6", "days_t8", "days_t9"]
READABLE = ["home", "workplace", "t1 (Eating)", "t3 (Attraction)", "t5 (Health)",
            "t6 (Infra)", "t8 (Retail)", "t9 (Transport)"]

print(f"[INFO] loading {DATA}")
df = pd.read_csv(DATA)
print(f"[INFO] {len(df):,} rows (already filtered + labelled)")


def feature_means_and_pca(df, year):
    cl = f"cluster_{year}"
    feats = [f"home_hours_{year}", f"workplace_days_{year}"] + [f"{c}_{year}" for c in BASE]
    means = df.groupby(cl)[feats].mean()
    means.columns = READABLE
    plt.figure(figsize=(10, 6))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"Original Feature Means by Cluster - {year}")
    fp = os.path.join(OUT, f"feature_means_original_values_{year}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[FIG] {fp}")

    ratio_cols = [f"home_hours_{year}_ratio", f"workplace_days_{year}_ratio"] + \
                 [f"{c}_{year}_ratio" for c in BASE]
    X = df[ratio_cols].values
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[cl], cmap="tab10", s=10, alpha=0.7)
    plt.title(f"PCA of Clusters (on Ratio Features) - {year}")
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.legend(*sc.legend_elements(), title="Clusters")
    fp = os.path.join(OUT, f"pca_clusters_{year}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[FIG] {fp}")


def transition(df):
    tm = pd.crosstab(df["cluster_2020"], df["cluster_2021"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(tm, annot=True, fmt="d", cmap="Blues")
    plt.title("Cluster Transition Matrix (2020 to 2021)")
    plt.xlabel("2021 Clusters (Aligned)"); plt.ylabel("2020 Clusters")
    fp = os.path.join(OUT, "transition_matrix_counts.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(); print(f"[FIG] {fp}")

    tp = (tm.div(tm.sum(axis=1), axis=0) * 100).round(2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(tp, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Cluster Transition Matrix (%) (2020 to 2021)")
    plt.xlabel("2021 Clusters (Aligned)"); plt.ylabel("2020 Clusters")
    fp = os.path.join(OUT, "transition_matrix_percent.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(); print(f"[FIG] {fp}")
    print(f"[INFO] overall change rate: {(df['cluster_2020'] != df['cluster_2021']).mean():.2%}")


def ternary_distribution(df, year):
    # verbatim from cluster_profiles.py: LAD-normalised days
    home_col = f"home_hours_{year}"; amen_col = f"hours_amenties_{year}_all"; work_col = f"workplace_hours_{year}"
    df[f"{home_col}_lad_mean"] = df.groupby("home_2021_lad_code")[f"home_days_{year}"].transform("mean")
    df[f"{amen_col}_lad_mean"] = df.groupby("home_2021_lad_code")[f"days_amenties_{year}_all"].transform("mean")
    df[f"{work_col}_lad_mean"] = df.groupby("home_2021_lad_code")[f"workplace_days_{year}"].transform("mean")
    df[f"{home_col}_norm"] = df[f"home_days_{year}"] / df[f"{home_col}_lad_mean"]
    df[f"{amen_col}_norm"] = df[f"days_amenties_{year}_all"] / df[f"{amen_col}_lad_mean"]
    df[f"{work_col}_norm"] = df[f"workplace_days_{year}"] / df[f"{work_col}_lad_mean"]
    data_points = df[[f"{home_col}_norm", f"{amen_col}_norm", f"{work_col}_norm"]].dropna().values

    # close each composition to sum 1, then add a small jitter so individuals
    # don't pile on top of each other (cosmetic de-overlap of the central cluster)
    data_points = data_points[data_points.sum(axis=1) > 0]
    pts = data_points / data_points.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(42)
    pts = np.clip(pts + rng.normal(0, 0.015, size=pts.shape), 1e-6, None)
    pts = pts / pts.sum(axis=1, keepdims=True)
    scale = 100

    fig, tax = ternary.figure(scale=scale)
    fig.set_size_inches(10, 8)
    tax.scatter((pts * scale).tolist(), marker="o", s=3, alpha=0.10,
                color="#2b6cb0", edgecolors="none", zorder=2)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=10)
    tax.ticks(axis="lbr", linewidth=1, multiple=20, offset=0.025, tick_formats="%.1f")
    tax.get_axes().axis("off"); tax.clear_matplotlib_ticks()
    tax.set_title(f"Home, Workplace, Amenities Distribution ({year})", fontsize=18)
    tax.bottom_axis_label("Home Days (%)", offset=0.06, fontsize=12)
    tax.right_axis_label("Amenities Days (%)", offset=0.16, fontsize=12)
    tax.left_axis_label("Workplace Days (%)", offset=0.16, fontsize=12)
    tax.get_axes().set_facecolor("white")
    fp = os.path.join(OUT, f"ternary_distribution_{year}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig); print(f"[FIG] {fp}")


for y in ["2020", "2021"]:
    feature_means_and_pca(df, y)
    ternary_distribution(df, y)
transition(df)
print("\n[DONE] figures in", OUT)
