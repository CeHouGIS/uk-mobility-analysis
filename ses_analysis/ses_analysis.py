#!/usr/bin/env python
"""
Socioeconomic heterogeneity of lifestyle clusters and behavioral inertia.
  - Clusters: reproduce the paper's KMeans (k=5, reference-aligned) from England_ML.csv
  - Home geography + income: final_analysis_data.csv (device -> home MSOA, income_2018)
  - Census SES (2021, MSOA): TS067 education (% Level 4+), TS062 NS-SeC (% managerial/professional)
  Analysis A: boxplots of SES by lifestyle cluster (+ Kruskal-Wallis)
  Analysis B: lifestyle-cluster stability (2020-02 -> 2021-03) by SES group (+ chi-square)
"""
import warnings; warnings.filterwarnings("ignore")
import os, io, zipfile
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ML = "/workplace/Mob/ML"
FA = "/workplace/Mob/fig_output/final_analysis_data.csv"
CEN = "/workplace/Mob/census_2021"
OUT = "/workplace/Mob/ses_analysis"; os.makedirs(OUT, exist_ok=True)
BASE = ['workplace_frequency', 't1_frequency', 't3_frequency', 't5_frequency',
        't6_frequency', 't8_frequency', 't9_frequency']
REF = np.array([[0.4,1.2,0.4,0.9,1.0,1.9,0.3],[0.6,4.7,1.5,3.0,3.3,6.5,0.8],
                [0.6,2.6,0.9,1.8,2.0,3.8,0.6],[0.8,1.2,0.4,0.9,0.9,1.9,0.2],
                [0.7,8.9,2.9,5.4,6.2,12.1,1.2]])


def get_clusters():
    df = pd.read_csv(f"{ML}/England_ML.csv")
    feb = df[df.month == 202002].copy(); mar = df[df.month == 202003].copy(); mar21 = df[df.month == 202103].copy()
    for c in BASE:
        feb[f"{c}_z"] = stats.zscore(feb[c]); mar[f"{c}_z"] = stats.zscore(mar[c]); mar21[f"{c}_z"] = stats.zscore(mar21[c])
    mar21.columns = ['device_aid'] + [c + '_202103' for c in mar21.columns[1:]]
    tem = feb.merge(mar, on='device_aid', how='inner', suffixes=('_202002', '_202003'))
    ds = tem.merge(mar21, on='device_aid', how='inner', suffixes=('', '_202103'))
    z = [f"{x}_z" for x in BASE]
    X02 = ds[[i + '_202002' for i in z]].copy(); X03 = ds[[i + '_202003' for i in z]].copy(); X13 = ds[[i + '_202103' for i in z]].copy()
    g = ['workplace', 'a1', 'a3', 'a5', 'a6', 'a8', 'a9']; X02.columns = X03.columns = X13.columns = g
    k = 5; lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pd.concat([X02, X03, X13])).labels_
    n2 = len(X02)
    ds['cluster_202002'] = lab[:n2]; ds['cluster_202003'] = lab[n2:2*n2]; ds['cluster_202103'] = lab[2*n2:]
    feb_cols = [f'{c}_202002' for c in BASE]
    mine = ds.groupby('cluster_202002')[feb_cols].mean().reindex(range(k)).values
    r, c = linear_sum_assignment(cdist(mine, REF)); remap = {int(a): int(b) for a, b in zip(r, c)}
    for col in ['cluster_202002', 'cluster_202003', 'cluster_202103']:
        ds[col] = ds[col].map(remap)
    print(f"[INFO] clustered devices: {len(ds):,}  | remap mine->ref {remap}")
    return ds[['device_aid', 'cluster_202002', 'cluster_202003', 'cluster_202103']]


def census_ses():
    z67 = zipfile.ZipFile(f"{CEN}/ts067.zip"); edu = pd.read_csv(io.BytesIO(z67.read("census2021-ts067-msoa.csv")))
    tcol = [c for c in edu.columns if 'Total' in c][0]
    edu['pct_degree'] = edu['Highest level of qualification: Level 4 qualifications and above'] / edu[tcol] * 100
    edu = edu[['geography code', 'pct_degree']].rename(columns={'geography code': 'msoa'})
    z62 = zipfile.ZipFile(f"{CEN}/ts062.zip"); occ = pd.read_csv(io.BytesIO(z62.read([n for n in z62.namelist() if 'msoa' in n][0])))
    tot = occ[[c for c in occ.columns if 'Total: All usual residents' in c][0]]
    hi = occ[[c for c in occ.columns if 'L1, L2 and L3' in c][0]]
    lo = occ[[c for c in occ.columns if 'L4, L5 and L6' in c][0]]
    occ['pct_manag_prof'] = (hi + lo) / tot * 100
    occ = occ[['geography code', 'pct_manag_prof']].rename(columns={'geography code': 'msoa'})
    return edu.merge(occ, on='msoa', how='outer')


# ---------------- join ----------------
clusters = get_clusters()
geo = pd.read_csv(FA, usecols=['device_aid', 'home_2021_msoa_code', 'income_2018']).rename(columns={'home_2021_msoa_code': 'msoa'})
geo['income_2018'] = pd.to_numeric(geo['income_2018'], errors='coerce')
ses = census_ses()
d = clusters.merge(geo, on='device_aid', how='left')
matched = d['msoa'].notna()
print(f"[INFO] clustered with home geo: {matched.sum():,} / {len(d):,} ({matched.mean():.1%})")
# selection check: cluster distribution matched vs unmatched
chk = pd.concat([d[matched]['cluster_202002'].value_counts(normalize=True).rename('matched'),
                 d[~matched]['cluster_202002'].value_counts(normalize=True).rename('unmatched')], axis=1).sort_index()
print("[INFO] cluster_202002 share (matched vs unmatched):\n", (chk * 100).round(1).to_string())
d = d[matched].merge(ses, on='msoa', how='left')

SES_VARS = [('income_2018', 'Neighbourhood median income (£)'),
            ('pct_degree', '% with degree (Level 4+)'),
            ('pct_manag_prof', '% managerial & professional')]
sns.set_context("paper"); sns.set_style("ticks")

# ---------------- Analysis A: SES by lifestyle cluster ----------------
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (var, lab) in zip(axs, SES_VARS):
    sub = d.dropna(subset=[var, 'cluster_202002'])
    sns.boxplot(data=sub, x='cluster_202002', y=var, ax=ax, showfliers=False, color='#5b9bd5')
    groups = [g[var].values for _, g in sub.groupby('cluster_202002')]
    H, p = stats.kruskal(*groups)
    ax.set_title(f"{lab}\nKruskal-Wallis H={H:.0f}, p={p:.1e}", fontsize=9)
    ax.set_xlabel('Lifestyle cluster (Feb 2020)'); ax.set_ylabel(lab)
sns.despine(); plt.tight_layout()
plt.savefig(f"{OUT}/A_ses_by_cluster.png", dpi=300, bbox_inches='tight'); plt.close()
print(f"[FIG] {OUT}/A_ses_by_cluster.png")

# cluster-level SES means table
print("\n[A] mean SES by lifestyle cluster (Feb 2020):")
print(d.groupby('cluster_202002')[[v for v, _ in SES_VARS]].mean().round(1).to_string())

# ---------------- Analysis B: stability by SES group ----------------
d['stable_2020'] = (d['cluster_202002'] == d['cluster_202003']).astype(int)   # Feb2020 -> Mar2020
d['stable_covid'] = (d['cluster_202002'] == d['cluster_202103']).astype(int)  # Feb2020 -> Mar2021
sub = d.dropna(subset=['income_2018']).copy()
sub['SES_group'] = pd.qcut(sub['income_2018'], 3, labels=['Low', 'Medium', 'High'])
rate = (sub.groupby('SES_group')[['stable_2020', 'stable_covid']].mean() * 100)
n_grp = sub.groupby('SES_group').size()
print("\n[B] stability rate (%) by income SES group:")
print(rate.round(1).assign(n=n_grp).to_string())
ct = pd.crosstab(sub['SES_group'], sub['stable_covid'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"[B] chi-square (SES x stable Feb2020->Mar2021): chi2={chi2:.1f}, dof={dof}, p={p:.2e}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
rate.rename(columns={'stable_2020': 'Feb 2020 → Mar 2020', 'stable_covid': 'Feb 2020 → Mar 2021'}).plot(
    kind='bar', ax=ax, color=['#9ecae1', '#3182bd'], width=0.8, edgecolor='black', linewidth=0.4)
ax.set_ylabel('Stayed in same lifestyle cluster (%)'); ax.set_xlabel('Neighbourhood income group')
ax.set_title(f"Lifestyle-cluster stability by SES\nchi-square (COVID transition) p={p:.1e}", fontsize=10)
ax.legend(frameon=False, title=None); plt.xticks(rotation=0)
sns.despine(); plt.tight_layout()
plt.savefig(f"{OUT}/B_stability_by_ses.png", dpi=300, bbox_inches='tight'); plt.close()
print(f"[FIG] {OUT}/B_stability_by_ses.png")

# save the joined analysis table (no raw device data beyond clusters+area SES)
d.to_csv(f"{OUT}/clusters_with_ses.csv", index=False)
print(f"\n[DONE] outputs in {OUT}")
