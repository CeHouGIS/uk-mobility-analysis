#!/usr/bin/env python
"""
KDE version of the Fig 3 ternary. Same data prep as triangle_plot.ipynb
(workplace / amenity / home frequency, each normalised to its mean, per period),
rendered as a smooth 2-D gaussian_kde density instead of raw grid counts.
The three periods share ONE common colour scale (vmax) so they are comparable.
Run from this directory:  cd /workplace/Mob/ML && python fig3_kde_ternary.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import ternary
from scipy.stats import gaussian_kde

SCALE = 100
SQRT3_2 = np.sqrt(3) / 2.0
PERIODS = [('202002', 'fig/ternary_kde_202002.jpg'),
           ('202003', 'fig/ternary_kde_202003.jpg'),
           ('202103', 'fig/ternary_kde_202103.jpg')]


def cart(comp):
    """barycentric (a=workplace, b=amenity, c=home) rows sum to 1 -> 2-D cartesian."""
    a, b, c = comp[:, 0], comp[:, 1], comp[:, 2]
    return np.vstack([b + 0.5 * c, c * SQRT3_2])


def build_df_select():
    df = pd.read_csv("England_ML 2.csv")
    df['amenity_frequency_all'] = df[['t1_frequency', 't3_frequency', 't5_frequency',
                                      't6_frequency', 't8_frequency', 't9_frequency']].sum(axis=1)
    feb = df[df['month'] == 202002]; mar = df[df['month'] == 202003]
    mar21 = df[df['month'] == 202103].copy()
    mar21.columns = ['device_aid'] + [c + '_202103' for c in mar21.columns[1:]]
    tem = feb.merge(mar, on='device_aid', how='inner', suffixes=('_202002', '_202003'))
    ds = tem.merge(mar21, on='device_aid', how='inner', suffixes=('', '_202103'))
    for m in ['202002', '202003', '202103']:
        ds[f'home_frequency_{m}_norm'] = ds[f'home_weighted_hours_today_{m}'] / ds[f'home_weighted_hours_today_{m}'].mean()
        ds[f'amenity_frequency_all_{m}_norm'] = ds[f'amenity_frequency_all_{m}'] / ds[f'amenity_frequency_all_{m}'].mean()
        ds[f'workplace_frequency_{m}_norm'] = ds[f'workplace_frequency_{m}'] / ds[f'workplace_frequency_{m}'].mean()
    return ds


def compute_density(df_select, period):
    pts = df_select[[f'workplace_frequency_{period}_norm',
                     f'amenity_frequency_all_{period}_norm',
                     f'home_frequency_{period}_norm']].dropna().values
    pts = pts[pts.sum(axis=1) > 0]
    pts = pts / pts.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    sample = pts if len(pts) <= 20000 else pts[rng.choice(len(pts), 20000, replace=False)]
    kde = gaussian_kde(cart(sample))
    keys, comp = [], []
    for i in range(SCALE + 1):
        for j in range(SCALE + 1 - i):
            keys.append((i, j, SCALE - i - j)); comp.append([i, j, SCALE - i - j])
    dens = kde(cart(np.array(comp) / SCALE))
    return {k: float(d) for k, d in zip(keys, dens)}, len(pts)


def plot_density(density, period, save, vmax):
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']; plt.rcParams['font.size'] = 12
    fig, tax = ternary.figure(scale=SCALE)
    fig.set_size_inches(5.5, 4.5)
    tax.heatmap(density, style="triangular", cmap='YlGnBu', colorbar=False, vmin=0, vmax=vmax)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='YlGnBu'); sm.set_array([])
    cb = plt.colorbar(sm, ax=tax.get_axes(), fraction=0.046, pad=0.04); cb.set_label('KDE density')
    tax.boundary(linewidth=0.8); tax.gridlines(color="lightgray", multiple=20, linewidth=0.5)
    tax.right_axis_label("Amenity frequency (%)", offset=0.12, fontsize=12)
    tax.bottom_axis_label("Workplace frequency (%)", offset=0.01, fontsize=12)
    tax.left_axis_label("At home time (%)", offset=0.12, fontsize=12)
    tax.ticks(axis='lbr', linewidth=0.5, multiple=20, tick_formats="%.0f")
    tax.get_axes().axis('off'); tax.clear_matplotlib_ticks()
    plt.savefig(save, bbox_inches='tight', dpi=300); plt.close(fig)


if __name__ == "__main__":
    ds = build_df_select()
    densities = {}
    for m, save in PERIODS:
        densities[m], n = compute_density(ds, m)
        print(f"[KDE] {m}: n={n:,}  99pct={np.percentile(list(densities[m].values()), 99):.2f}")
    # shared colour scale: max of the per-period 99th percentiles (robust + consistent)
    vmax = max(np.percentile(list(d.values()), 99) for d in densities.values())
    print(f"[INFO] shared vmax = {vmax:.2f}")
    for m, save in PERIODS:
        plot_density(densities[m], m, save, vmax)
        print(f"[FIG] {save}")
    print("[DONE]")
