#!/usr/bin/env python
"""
Regenerate mobility figures from the long-format weighted-hours/frequency table
(/workplace/Mob/data/England_ML.csv): one row per (device_aid, month).

Two groups of output:
  1. FAITHFUL  — directly supported by this data:
       - Ternary Home/Workplace/Amenities distribution per period
       - Mean weighted-hours / frequency heatmaps (place x period)
       - Boxplots of weighted hours by period
  2. ADAPTED CLUSTERING — analog of the repo's cluster_profiles.py figures, but
       clustered DIRECTLY on the 8 weighted-hours features (NO LAD normalization,
       periods instead of calendar years). Clearly a different methodology.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

DATA = "/workplace/Mob/data/England_ML.csv"
OUT = "/workplace/Mob/mobility_figures"
OUT_CLU = os.path.join(OUT, "adapted_clustering")
os.makedirs(OUT_CLU, exist_ok=True)

PERIOD_LABEL = {202002: "Feb 2020", 202003: "Mar 2020", 202103: "Mar 2021", 202203: "Mar 2022"}
PERIOD_ORDER = [202002, 202003, 202103, 202203]
WH = "_weighted_hours_today"
AMEN = ["t1", "t3", "t5", "t6", "t8", "t9"]
AMEN_LABEL = {"t1": "t1 Eating/Drink", "t3": "t3 Attractions", "t5": "t5 Health",
              "t6": "t6 Public Infra", "t8": "t8 Retail", "t9": "t9 Transport"}
PLACE_ORDER = ["home", "workplace"] + AMEN
PLACE_LABEL = {"home": "Home", "workplace": "Workplace", **AMEN_LABEL}

print(f"[INFO] Loading {DATA} ...")
df = pd.read_csv(DATA)
df = df.drop_duplicates(["device_aid", "month"])
df["amenity" + WH] = df[[a + WH for a in AMEN]].sum(axis=1)
df["amenity_frequency"] = df[[a + "_frequency" for a in AMEN]].sum(axis=1)
print(f"[INFO] {len(df):,} rows | {df['device_aid'].nunique():,} devices | months {sorted(df['month'].unique())}")

manifest = []

# ---------------------------------------------------------------- 1. TERNARY
def ternary_plot(sub, period, path):
    home = sub["home" + WH].to_numpy()
    work = sub["workplace" + WH].to_numpy()
    amen = sub["amenity" + WH].to_numpy()
    total = home + work + amen
    keep = total > 0
    home, work, amen, total = home[keep], work[keep], amen[keep], total[keep]
    a, b, c = home / total, amen / total, work / total          # bottom, right, left
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
    tax.set_title(f"Home / Workplace / Amenities time share — {PERIOD_LABEL[period]}\n"
                  f"(n={keep.sum():,} devices)", fontsize=15)
    tax.bottom_axis_label("Home share", offset=0.06, fontsize=12)
    tax.right_axis_label("Amenities share", offset=0.16, fontsize=12)
    tax.left_axis_label("Workplace share", offset=0.16, fontsize=12)
    tax.ticks(axis="lbr", linewidth=1, multiple=20, offset=0.025, tick_formats="%.1f")
    tax.get_axes().axis("off")
    tax.clear_matplotlib_ticks()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

for p in PERIOD_ORDER:
    sub = df[df["month"] == p]
    if sub.empty:
        continue
    fp = os.path.join(OUT, f"ternary_{p}.png")
    ternary_plot(sub, p, fp)
    manifest.append(fp)
    print(f"[FIG] {fp}")

# ----------------------------------------------- 2. HEATMAPS (place x period)
def place_period_matrix(suffix):
    cols = ["home", "workplace"] + AMEN
    rows = {}
    for p in PERIOD_ORDER:
        sub = df[df["month"] == p]
        if sub.empty:
            continue
        rows[PERIOD_LABEL[p]] = [sub[c + suffix].mean() for c in cols]
    m = pd.DataFrame(rows, index=[PLACE_LABEL[c] for c in cols])
    return m

for suffix, name, title in [(WH, "weighted_hours", "Mean weighted hours / day"),
                            ("_frequency", "frequency", "Mean visit frequency")]:
    m = place_period_matrix(suffix)
    plt.figure(figsize=(7, 6))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"{title} by place and period")
    plt.ylabel("Place type"); plt.xlabel("Period")
    fp = os.path.join(OUT, f"{name}_heatmap.png")
    plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
    manifest.append(fp); print(f"[FIG] {fp}")

# grouped bar of the same weighted-hours matrix (easier to read trends)
m = place_period_matrix(WH)
ax = m.T.plot(kind="bar", figsize=(11, 6), width=0.8, colormap="tab10")
ax.set_title("Mean weighted hours per day by place type, across periods")
ax.set_ylabel("Mean weighted hours / day"); ax.set_xlabel("Period")
ax.legend(title="Place", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.xticks(rotation=0)
fp = os.path.join(OUT, "mean_weighted_hours_by_period.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

# ----------------------------------------------- 3. BOXPLOTS (distribution)
long = []
for p in PERIOD_ORDER:
    sub = df[df["month"] == p]
    if sub.empty:
        continue
    for place in ["home", "workplace", "amenity"]:
        v = sub[place + WH].clip(upper=24)
        long.append(pd.DataFrame({"period": PERIOD_LABEL[p], "place": place.title(),
                                  "weighted_hours": v}))
long = pd.concat(long, ignore_index=True)
plt.figure(figsize=(11, 6))
sns.boxplot(data=long, x="place", y="weighted_hours", hue="period", showfliers=False)
plt.title("Distribution of weighted hours by place type and period")
plt.ylabel("Weighted hours / day (capped at 24)"); plt.xlabel("")
plt.legend(title="Period")
fp = os.path.join(OUT, "weighted_hours_boxplot_by_period.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

# ===================================================== 4. ADAPTED CLUSTERING
# Cluster devices on the 8 weighted-hours features, independently for Mar 2020
# and Mar 2021, align labels (Hungarian), and compare. NOT the repo methodology.
FEATS = ["home" + WH, "workplace" + WH] + [a + WH for a in AMEN]
READABLE = [PLACE_LABEL[c] for c in PLACE_ORDER]
K = 7
P1, P2 = 202003, 202103

d1 = df[df["month"] == P1].set_index("device_aid")[FEATS].fillna(0)
d2 = df[df["month"] == P2].set_index("device_aid")[FEATS].fillna(0)
common = d1.index.intersection(d2.index)
X1, X2 = d1.loc[common], d2.loc[common]
print(f"[INFO] adapted clustering on {len(common):,} devices present in both "
      f"{PERIOD_LABEL[P1]} & {PERIOD_LABEL[P2]}")

def fit(X):
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs), Xs

km1, Xs1 = fit(X1)
km2, Xs2 = fit(X2)
# align period-2 labels to period-1
cost = cdist(km1.cluster_centers_, km2.cluster_centers_)
r, c = linear_sum_assignment(cost)
remap = {old: new for new, old in zip(r, c)}
lab1 = km1.labels_
lab2 = np.array([remap[l] for l in km2.labels_])

# 4a. feature-means heatmap (period 1)
cm = X1.copy(); cm["cluster"] = lab1
means = cm.groupby("cluster")[FEATS].mean(); means.columns = READABLE
plt.figure(figsize=(10, 6))
sns.heatmap(means, annot=True, fmt=".2f", cmap="Blues")
plt.title(f"[ADAPTED] Cluster mean weighted hours — {PERIOD_LABEL[P1]} (k={K})")
plt.xlabel("Place type"); plt.ylabel("Cluster")
fp = os.path.join(OUT_CLU, f"cluster_feature_means_{P1}.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

# 4b. PCA scatter (period 1)
pca = PCA(n_components=2).fit_transform(Xs1)
plt.figure(figsize=(8, 6))
sc = plt.scatter(pca[:, 0], pca[:, 1], c=lab1, cmap="tab10", s=6, alpha=0.5)
plt.title(f"[ADAPTED] PCA of clusters — {PERIOD_LABEL[P1]}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(*sc.legend_elements(), title="Cluster", fontsize=8)
fp = os.path.join(OUT_CLU, f"pca_clusters_{P1}.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

# 4c. transition matrix P1 -> P2
tm = pd.crosstab(pd.Series(lab1, name=PERIOD_LABEL[P1]),
                 pd.Series(lab2, name=f"{PERIOD_LABEL[P2]} (aligned)"))
plt.figure(figsize=(8, 6))
sns.heatmap(tm, annot=True, fmt="d", cmap="Blues")
plt.title(f"[ADAPTED] Cluster transition counts ({PERIOD_LABEL[P1]} -> {PERIOD_LABEL[P2]})")
fp = os.path.join(OUT_CLU, "transition_matrix_counts.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")

tmp = tm.div(tm.sum(axis=1), axis=0) * 100
plt.figure(figsize=(8, 6))
sns.heatmap(tmp.round(1), annot=True, fmt=".1f", cmap="Blues")
plt.title(f"[ADAPTED] Cluster transition % ({PERIOD_LABEL[P1]} -> {PERIOD_LABEL[P2]})")
fp = os.path.join(OUT_CLU, "transition_matrix_percent.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
manifest.append(fp); print(f"[FIG] {fp}")
change = (lab1 != lab2).mean()
print(f"[INFO] adapted overall cluster change rate {PERIOD_LABEL[P1]}->{PERIOD_LABEL[P2]}: {change:.1%}")

print("\n[DONE] figures written:")
for m in manifest:
    print("   ", m)
print(f"\nTotal: {len(manifest)} figures")
