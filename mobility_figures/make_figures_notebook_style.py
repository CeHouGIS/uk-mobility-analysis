#!/usr/bin/env python
"""
Prettier figures, with plotting styling ported from the notebooks
  - notebooks/s4_ternary_plots.ipynb  (ternary heatmap + scatter-by-cluster)
  - notebooks/s4_clustering.ipynb      (elbow plot + RdBu_r centered heatmap)

Driven by the long-format weighted-hours table /workplace/Mob/data/England_ML.csv
(one row per device x month). The notebooks originally read a *wide* per-individual
table with LAD-normalised day/hour columns; that table isn't available here, so the
ternary uses raw weighted-hours shares (home / workplace / amenities) per period.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import ternary
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

DATA = "/workplace/Mob/data/England_ML.csv"
OUT = "/workplace/Mob/mobility_figures/notebook_style"
os.makedirs(OUT, exist_ok=True)

PERIOD_LABEL = {202002: "Feb 2020", 202003: "Mar 2020", 202103: "Mar 2021", 202203: "Mar 2022"}
PERIOD_ORDER = [202002, 202003, 202103, 202203]
WH = "_weighted_hours_today"
AMEN = ["t1", "t3", "t5", "t6", "t8", "t9"]
FEATS = ["home" + WH, "workplace" + WH] + [a + WH for a in AMEN]
READABLE = ["Home", "Workplace", "t1 Eat/Drink", "t3 Attractions", "t5 Health",
            "t6 Public Infra", "t8 Retail", "t9 Transport"]

print(f"[INFO] loading {DATA}")
df = pd.read_csv(DATA).drop_duplicates(["device_aid", "month"])
df["amenity" + WH] = df[[a + WH for a in AMEN]].sum(axis=1)
manifest = []


# --------------------------------------------------------------------------
# Ported from s4_ternary_plots.ipynb  (cells 2/4): ternary heatmap styling
# --------------------------------------------------------------------------
def pretty_ternary_heatmap(home, work, amen, title, path):
    total = home + work + amen
    keep = total > 0
    home, work, amen = home[keep], work[keep], amen[keep]
    a = home / (home + work + amen)          # bottom = Home
    b = amen / (home + work + amen)          # right  = Amenities
    scale = 100
    counts = {}
    for ai, bi in zip(a, b):
        i = int(round(ai * scale)); j = int(round(bi * scale)); k = scale - i - j
        if k < 0:
            continue
        counts[(i, j, k)] = counts.get((i, j, k), 0) + 1

    fig, tax = ternary.figure(scale=scale)
    fig.set_size_inches(10, 8)
    tax.heatmap(counts, style="triangular", cmap="YlGnBu", colorbar=True)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=10)
    tax.gridlines(color="blue", multiple=1, linewidth=0.5, alpha=0.1)
    tax.ticks(axis="lbr", linewidth=1, multiple=20, offset=0.025, tick_formats="%.1f")
    tax.get_axes().axis("off")
    tax.clear_matplotlib_ticks()
    tax.set_title(title, fontsize=18)
    tax.left_axis_label("Workplace share (%)", offset=0.16, fontsize=12)
    tax.right_axis_label("Amenities share (%)", offset=0.16, fontsize=12)
    tax.bottom_axis_label("Home share (%)", offset=0.06, fontsize=12)
    tax.get_axes().set_facecolor("white")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


for p in PERIOD_ORDER:
    sub = df[df["month"] == p]
    if sub.empty:
        continue
    fp = os.path.join(OUT, f"ternary_heatmap_{p}.png")
    pretty_ternary_heatmap(sub["home" + WH].to_numpy(), sub["workplace" + WH].to_numpy(),
                           sub["amenity" + WH].to_numpy(),
                           f"Home, Workplace, Amenities Distribution ({PERIOD_LABEL[p]})", fp)
    manifest.append(fp); print(f"[FIG] {fp}")


# --------------------------------------------------------------------------
# Adapted clustering (Mar 2020) to drive the cluster-coloured figures below
# --------------------------------------------------------------------------
P1 = 202003
X = df[df["month"] == P1].set_index("device_aid")[FEATS].fillna(0)
Xs = StandardScaler().fit_transform(X)

# Elbow plot  -- ported from s4_clustering.ipynb cell 6
K_range = range(2, 11)
sse = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs).inertia_ for k in K_range]
elbow_k = 7
plt.figure(figsize=(6, 4))
plt.plot(list(K_range), sse, marker="o")
plt.scatter(elbow_k, sse[list(K_range).index(elbow_k)], color="red", s=100, label=f"chosen k={elbow_k}")
plt.xlabel("k"); plt.ylabel("SSE (inertia)"); plt.title(f"Elbow — {PERIOD_LABEL[P1]}")
plt.legend(); plt.tight_layout()
fp = os.path.join(OUT, f"elbow_{P1}.png")
plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

K = 7
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs)
labels = km.labels_

# Cluster-means heatmap (standardized, centered) -- ported from s4_clustering cell 13
cm = pd.DataFrame(Xs, columns=READABLE); cm["cluster"] = labels
means = cm.groupby("cluster")[READABLE].mean()
plt.figure(figsize=(10, 6))
sns.heatmap(means, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title(f"Cluster mean (standardized) — {PERIOD_LABEL[P1]} (k={K})")
plt.xlabel("Place type"); plt.ylabel("Cluster")
fp = os.path.join(OUT, f"cluster_means_standardized_{P1}.png")
plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

# Ternary SCATTER coloured by cluster -- ported from s4_ternary_plots.ipynb cell 7
sub = df[df["month"] == P1].set_index("device_aid").loc[X.index]
home, work, amen = sub["home" + WH].to_numpy(), sub["workplace" + WH].to_numpy(), sub["amenity" + WH].to_numpy()
total = home + work + amen
keep = total > 0
pts = np.column_stack([home[keep], amen[keep], work[keep]])
pts = pts / pts.sum(axis=1, keepdims=True) * 100      # bottom=Home, right=Amenities, left=Workplace
lab = labels[keep]
cmap = plt.get_cmap("tab10")
norm = mcolors.Normalize(vmin=lab.min(), vmax=lab.max())
colors = cmap(norm(lab))
scale = 100
fig, tax = ternary.figure(scale=scale)
fig.set_size_inches(10, 8)
tax.scatter(pts.tolist(), marker="o", color=colors, s=3, edgecolors="k", linewidth=0, zorder=3)
tax.boundary(linewidth=2.0)
tax.gridlines(color="grey", multiple=10, linewidth=0.5)
tax.set_title(f"Ternary scatter coloured by cluster — {PERIOD_LABEL[P1]}", fontsize=16, y=1.03)
tax.left_axis_label("Workplace share (%)", fontsize=12, offset=0.14)
tax.right_axis_label("Amenities share (%)", fontsize=12, offset=0.14)
tax.bottom_axis_label("Home share (%)", fontsize=12, offset=0.14)
tax.ticks(axis="lbr", linewidth=1, multiple=10, offset=0.025, fontsize=10)
tax.get_axes().axis("off"); tax.clear_matplotlib_ticks()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = plt.colorbar(sm, ax=tax.get_axes(), shrink=0.8, aspect=20, pad=0.05)
cbar.set_label("Cluster", fontsize=12, labelpad=10)
tax.get_axes().set_facecolor("white")
fp = os.path.join(OUT, f"ternary_scatter_cluster_{P1}.png")
plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
manifest.append(fp); print(f"[FIG] {fp}")

print(f"\n[DONE] {len(manifest)} figures in {OUT}:")
for m in manifest:
    print("   ", os.path.basename(m))
