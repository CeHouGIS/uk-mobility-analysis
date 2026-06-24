#!/usr/bin/env python
"""
KDE version of the faithful ternary: same LAD-normalised composition as
redraw_from_final_analysis.py, but rendered as a smooth 2-D kernel-density
estimate (gaussian_kde in the triangle's cartesian space) instead of raw
grid counts. Output: a separate set in ternary_kde/ (faithful figures untouched).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ternary
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

DATA = "/workplace/Mob/fig_output/final_analysis_data.csv"
OUT = "/workplace/Mob/mobility_figures/ternary_kde"
os.makedirs(OUT, exist_ok=True)
SCALE = 100
SQRT3_2 = np.sqrt(3) / 2.0

print(f"[INFO] loading {DATA}")
df = pd.read_csv(DATA)


def cart(comp):
    """barycentric (home, amen, work), rows sum to 1 -> 2-D cartesian."""
    a, b, c = comp[:, 0], comp[:, 1], comp[:, 2]
    return np.vstack([b + 0.5 * c, c * SQRT3_2])


def lad_norm_composition(df, year):
    # identical data prep to the faithful LAD ternary
    hm = df.groupby("home_2021_lad_code")[f"home_days_{year}"].transform("mean")
    am = df.groupby("home_2021_lad_code")[f"days_amenties_{year}_all"].transform("mean")
    wm = df.groupby("home_2021_lad_code")[f"workplace_days_{year}"].transform("mean")
    P = pd.DataFrame({
        "home": df[f"home_days_{year}"] / hm,
        "amen": df[f"days_amenties_{year}_all"] / am,
        "work": df[f"workplace_days_{year}"] / wm,
    }).dropna().values
    P = P[P.sum(axis=1) > 0]
    return P / P.sum(axis=1, keepdims=True)


def kde_ternary(P, year, path):
    rng = np.random.default_rng(0)
    sample = P if len(P) <= 20000 else P[rng.choice(len(P), 20000, replace=False)]
    kde = gaussian_kde(cart(sample))           # fit 2-D KDE (Scott's-rule bandwidth)

    keys, comp = [], []
    for i in range(SCALE + 1):
        for j in range(SCALE + 1 - i):
            keys.append((i, j, SCALE - i - j))
            comp.append([i, j, SCALE - i - j])
    dens = kde(cart(np.array(comp) / SCALE))    # one vectorised evaluation
    density = {k: float(d) for k, d in zip(keys, dens)}

    fig, tax = ternary.figure(scale=SCALE)
    fig.set_size_inches(10, 8)
    tax.heatmap(density, style="triangular", cmap="YlGnBu", colorbar=True)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=10)
    tax.ticks(axis="lbr", linewidth=1, multiple=20, offset=0.025, tick_formats="%.1f")
    tax.get_axes().axis("off"); tax.clear_matplotlib_ticks()
    tax.set_title(f"Home, Workplace, Amenities — KDE density ({year})", fontsize=16)
    tax.bottom_axis_label("Home Days (%)", offset=0.06, fontsize=12)
    tax.right_axis_label("Amenities Days (%)", offset=0.16, fontsize=12)
    tax.left_axis_label("Workplace Days (%)", offset=0.16, fontsize=12)
    tax.get_axes().set_facecolor("white")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


for y in ["2020", "2021"]:
    P = lad_norm_composition(df, y)
    fp = os.path.join(OUT, f"ternary_kde_{y}.png")
    kde_ternary(P, y, fp)
    print(f"[FIG] {fp}  (n={len(P):,})")
print("[DONE]", OUT)
