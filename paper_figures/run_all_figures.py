#!/usr/bin/env python
"""
Faithful reproduction of the figures in /workplace/Mob/ML/ (ML.ipynb + triangle_plot.ipynb).
Run from this directory (CWD) so the relative 'fig/' and 'graph_data/' paths resolve.
  Fig 1  binscatter        <- Englnad_ML_merge_binscatter.csv  (wide)
  Fig 2  coefficient plot  <- graph_data/Figure_2*_data.csv
  Fig 3  ternary           <- England_ML 2.csv                 (long)
  Fig 4  cluster heatmaps  <- England_ML.csv                   (long)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def set_nature_params():
    sns.set_context("paper"); sns.set_style("ticks")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 1, 'lines.markersize': 4, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


# ======================= Fig 1: binscatter =======================
def get_binscatter_data(data, x_col, y_col, n_bins=20):
    data = data.copy()
    data['bin'] = pd.qcut(data[x_col], q=n_bins, duplicates='drop')
    bin_stats = data.groupby('bin', observed=True).agg({x_col: 'mean', y_col: ['mean', 'sem']})
    bin_stats.columns = ['x_mean', 'y_mean', 'y_sem']
    return bin_stats.reset_index()


def plot_fig1a_nature_style(df, t_type, savename):
    set_nature_params()
    stats_2020 = get_binscatter_data(df, f'{t_type}_freq_2020_02', f'{t_type}_freq_2020_03', n_bins=20)
    stats_2021 = get_binscatter_data(df, f'{t_type}_freq_2020_02', f'{t_type}_freq_2021_03', n_bins=20)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = ['#D55E00', '#00468B']
    amenity_dict = {1: 'A1\n(Eating & Drinking)', 3: 'A2 (Attractions)', 5: 'A3 (Health)',
                    6: 'A4 (Public\n Infrastructure)', 8: 'A5 (Retail)', 9: 'A6 (Transport)'}
    ax.scatter(stats_2020['x_mean'], stats_2020['y_mean'], marker='o', color=colors[0],
               s=20, edgecolor='white', linewidth=0.5, zorder=5)
    ax.scatter(stats_2021['x_mean'], stats_2021['y_mean'], marker='o', color=colors[1],
               s=20, edgecolor='white', linewidth=0.5, zorder=5)
    for y_col, color, label, ls in zip([f'{t_type}_freq_2020_03', f'{t_type}_freq_2021_03'],
                                        colors, ['2020-03', '2021-03'], ['-', '--']):
        sns.regplot(data=df, x=f'{t_type}_freq_2020_02', y=y_col, scatter=False, order=2,
                    ax=ax, color=color, label=label,
                    line_kws={'linewidth': 1.2, 'alpha': 0.8, 'linestyle': ls})
    ax.set_xlabel('Frequency (February 2020)', fontsize=8)
    ax.set_ylabel('Frequency (March 2020 & 2021)', fontsize=8)
    ax.set_title(f"Binscatter of Visits Frequencies for {amenity_dict[int(t_type[-1])]}", fontsize=9, pad=10)
    ax.legend(frameon=False, loc='upper left')
    x_max = stats_2021['x_mean'].max(); x_min = stats_2020['x_mean'].min()
    pad = (x_max - x_min) * 0.05
    ax.set_xlim(max(-0.2, x_min - pad) if x_min - pad < -0.1 else -0.2, x_max + pad)
    y_max = stats_2021['y_mean'].max(); y_min = stats_2020['y_mean'].min()
    pad_y = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    sns.despine(); plt.tight_layout()
    plt.savefig(f'fig/{savename}', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig1():
    df = pd.read_csv('Englnad_ML_merge_binscatter.csv')
    for t, name in [('t1', 'Fig 1a t1_binscatter.png'), ('t3', 'Fig 1b t3_binscatter.png'),
                    ('t5', 'Fig 1c t5_binscatter.png'), ('t6', 'Fig 1d t6_binscatter.png'),
                    ('t8', 'Fig 1e t8_binscatter.png'), ('t9', 'Fig 1f t9_binscatter.png')]:
        plot_fig1a_nature_style(df, t, name); print(f"[FIG] fig/{name}")


# ======================= Fig 2: coefficient plots =======================
def plot_fig2_nature_style(df_name, xlabel, savename):
    fig_2a = pd.read_csv(f"graph_data/{df_name}")
    set_nature_params()
    fig, ax = plt.subplots(figsize=(4, 3))
    yerr_low = fig_2a['coef'] - fig_2a['ci_low']; yerr_high = fig_2a['ci_high'] - fig_2a['coef']
    if 'model' not in fig_2a.columns:
        fig_2a['model'] = fig_2a['term'].apply(lambda x: x.split('::')[-1])
    ax.errorbar(x=fig_2a['model'], y=fig_2a['coef'], yerr=[yerr_low, yerr_high], fmt='o',
                color='black', ecolor='black', elinewidth=0.8, capsize=3, capthick=0.8,
                markersize=1, label='Estimate')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    if 'term' not in fig_2a.columns:
        ax.set_ylim(0, fig_2a['ci_high'].max() * 1.1); ax.set_xlabel('')
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Coefficient estimate (95% CI)', labelpad=10)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True); sns.despine(); plt.tight_layout()
    plt.savefig(f'fig/{savename}', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig2():
    for n, xl, save in [("Figure_2a_data.csv", None, "Fig 2a coefficient_plot.png"),
                        ("Figure_2b_data.csv", None, "Fig 2b coefficient_plot.png"),
                        ("Figure_2c_data.csv", 'Frequency quantile', "Fig 2c coefficient_plot.png"),
                        ("Figure_2d_data.csv", 'Hours quantile', "Fig 2d coefficient_plot.png")]:
        plot_fig2_nature_style(n, xl, save); print(f"[FIG] fig/{save}")


# ======================= Fig 3: ternary =======================
def fig3():
    df = pd.read_csv("England_ML 2.csv")
    df['amenity_frequency_all'] = df[['t1_frequency', 't3_frequency', 't5_frequency',
                                      't6_frequency', 't8_frequency', 't9_frequency']].sum(axis=1)
    df_2020_feb = df[df['month'] == 202002]; df_2020_mar = df[df['month'] == 202003]
    df_2021_mar = df[df['month'] == 202103].copy()
    df_2021_mar.columns = ['device_aid'] + [c + '_202103' for c in df_2021_mar.columns[1:]]
    tem = df_2020_feb.merge(df_2020_mar, on='device_aid', how='inner', suffixes=('_202002', '_202003'))
    df_select = tem.merge(df_2021_mar, on='device_aid', how='inner', suffixes=('', '_202103'))
    for m in ['202002', '202003', '202103']:
        df_select[f'home_frequency_{m}_norm'] = df_select[f'home_weighted_hours_today_{m}'] / df_select[f'home_weighted_hours_today_{m}'].mean()
        df_select[f'amenity_frequency_all_{m}_norm'] = df_select[f'amenity_frequency_all_{m}'] / df_select[f'amenity_frequency_all_{m}'].mean()
        df_select[f'workplace_frequency_{m}_norm'] = df_select[f'workplace_frequency_{m}'] / df_select[f'workplace_frequency_{m}'].mean()

    import ternary
    for m, save in [('202002', 'Fig 3a ternary_distribution_202002.jpg'),
                    ('202003', 'Fig 3b ternary_distribution_202003.jpg'),
                    ('202103', 'Fig 3c ternary_distribution_202103.jpg')]:
        data_points = df_select[[f'workplace_frequency_{m}_norm', f'amenity_frequency_all_{m}_norm',
                                 f'home_frequency_{m}_norm']].dropna().values
        scale = 100; counts = {}
        for (a, b, c) in data_points:
            total = a + b + c
            if total == 0:
                continue
            a, b, c = a / total, b / total, c / total
            i = int(round(a * scale)); j = int(round(b * scale)); k = scale - i - j
            if k < 0:
                continue
            counts[(i, j, k)] = counts.get((i, j, k), 0) + 1
        plt.rcParams['font.sans-serif'] = ['Arial']; plt.rcParams['font.size'] = 12
        fig, tax = ternary.figure(scale=scale)
        fig.set_size_inches(5.5, 4.5)
        tax.heatmap(counts, style="triangular", cmap='YlGnBu', colorbar=False)
        norm = mpl.colors.Normalize(vmin=0, vmax=600)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap='YlGnBu'); mappable.set_array([])
        plt.colorbar(mappable, ax=tax.get_axes(), fraction=0.046, pad=0.04)
        tax.boundary(linewidth=0.8); tax.gridlines(color="lightgray", multiple=20, linewidth=0.5)
        tax.right_axis_label("Amenity frequency (%)", offset=0.12, fontsize=12)
        tax.bottom_axis_label("Workplace frequency (%)", offset=0.01, fontsize=12)
        tax.left_axis_label("At home time (%)", offset=0.12, fontsize=12)
        tax.ticks(axis='lbr', linewidth=0.5, multiple=20, tick_formats="%.0f")
        tax.get_axes().axis('off'); tax.clear_matplotlib_ticks()
        plt.savefig(f'fig/{save}', bbox_inches='tight', dpi=300); plt.close(fig)
        print(f"[FIG] fig/{save}")


# ======================= Fig 4: clustering + heatmaps =======================
def fig4():
    from sklearn.cluster import KMeans
    BASE_FREQ_COLS = ['workplace_frequency', 't1_frequency', 't3_frequency', 't5_frequency',
                      't6_frequency', 't8_frequency', 't9_frequency']
    df = pd.read_csv("England_ML.csv")
    df_2020_feb = df[df['month'] == 202002].copy(); df_2020_mar = df[df['month'] == 202003].copy()
    df_2021_mar = df[df['month'] == 202103].copy()
    for col in BASE_FREQ_COLS:
        df_2020_feb[f"{col}_zscore"] = stats.zscore(df_2020_feb[col])
        df_2020_mar[f"{col}_zscore"] = stats.zscore(df_2020_mar[col])
        df_2021_mar[f"{col}_zscore"] = stats.zscore(df_2021_mar[col])
    df_2021_mar.columns = ['device_aid'] + [c + '_202103' for c in df_2021_mar.columns[1:]]
    tem = df_2020_feb.merge(df_2020_mar, on='device_aid', how='inner', suffixes=('_202002', '_202003'))
    df_select = tem.merge(df_2021_mar, on='device_aid', how='inner', suffixes=('', '_202103'))

    zcols = [f"{x}_zscore" for x in BASE_FREQ_COLS]
    X_202002 = df_select[[i + '_202002' for i in zcols]].copy()
    X_202003 = df_select[[i + '_202003' for i in zcols]].copy()
    X_202103 = df_select[[i + '_202103' for i in zcols]].copy()
    generic = ['workplace', 'a1', 'a3', 'a5', 'a6', 'a8', 'a9']
    X_202002.columns = X_202003.columns = X_202103.columns = generic
    X_scaled = pd.concat([X_202002, X_202003, X_202103], axis=0)

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    lab = kmeans.labels_
    n2 = len(X_202002)
    df_select['cluster_202002'] = lab[:n2]
    df_select['cluster_202003'] = lab[n2:2 * n2]
    df_select['cluster_202103'] = lab[2 * n2:]

    # Fig 4d: Feb 2020 -> Mar 2020
    tm = pd.crosstab(df_select['cluster_202002'], df_select['cluster_202003'])
    _transition_heatmap(tm, 'Cluster (Mar 2020)', 'fig/Fig 4d heatmap_nature.jpg')
    # Fig 4e: Feb 2020 -> Mar 2021
    tm2 = pd.crosstab(df_select['cluster_202002'], df_select['cluster_202103'])
    _transition_heatmap(tm2, 'Cluster (Mar 2021)', 'fig/Fig 4e heatmap_nature.jpg')

    # Fig 4a/4b/4c: per-period cluster mean-frequency heatmaps
    for m, save in [('202002', 'fig/Fig 4a Cluster_heatmap_202002.png'),
                    ('202003', 'fig/Fig 4b Cluster_heatmap_202003.png'),
                    ('202103', 'fig/Fig 4c Cluster_heatmap_202103.png')]:
        _cluster_heatmap(df_select, m, save)


def _transition_heatmap(transition_matrix, xlabel, save):
    set_nature_params()
    plt.figure(figsize=(3.5, 2.8))
    tp = (transition_matrix.div(transition_matrix.sum(axis=1), axis=0) * 100).round(2)
    sns.heatmap(tp, annot=True, fmt=".1f", cmap="Blues", linewidths=0, cbar_kws={'label': 'Probability (%)'})
    plt.xlabel(xlabel, fontsize=9); plt.ylabel('Cluster (Feb 2020)', fontsize=9)
    plt.tight_layout()
    plt.savefig(save, format='jpg', dpi=600, bbox_inches='tight'); plt.close()
    print(f"[FIG] {save}")


def _cluster_heatmap(df, year, save):
    feature_cols = [f'workplace_frequency_{year}', f't1_frequency_{year}', f't3_frequency_{year}',
                    f't5_frequency_{year}', f't6_frequency_{year}', f't8_frequency_{year}', f't9_frequency_{year}']
    readable = ["Workplace", 'A1 (Eating & Drinking)', 'A2 (Attractions)', 'A3 (Health)',
                'A4 (Public Infrastructure)', 'A5 (Retail)', 'A6 (Transport)']
    cluster_means = df.groupby(f'cluster_{year}')[feature_cols].mean()
    cluster_means.columns = readable
    set_nature_params()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax = sns.heatmap(cluster_means, annot=True, fmt=".1f", cmap="Blues", square=True,
                     linewidths=0, cbar_kws={'label': 'Probability (%)'})
    titles = {'202002': 'Mean Visitation Frequencies (Feb 2020)',
              '202003': 'Mean Visitation Frequencies (Mar 2020)',
              '202103': 'Mean Visitation Frequencies (Mar 2021)'}
    ax.set_title(titles[year]); ax.set_ylabel('Cluster ID')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(save, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[FIG] {save}")


if __name__ == "__main__":
    print("--- Fig 1 binscatter ---"); fig1()
    print("--- Fig 2 coefficient ---"); fig2()
    print("--- Fig 3 ternary ---"); fig3()
    print("--- Fig 4 clustering ---"); fig4()
    print("\n[DONE]")
