#!/usr/bin/env python
"""
Socioeconomic heterogeneity of lifestyle clusters and behavioral inertia.
Four neighbourhood (MSOA-level) dimensions:
  income           - income_2018 aggregated to MSOA mean (native value is ~LSOA-level)
  occupational     - pct_manag_prof : Census 2021 TS062 NS-SeC, % managerial & professional
  demographic/age  - median_age     : Census 2021 TS007a (from 5-year bands)
  demographic/eth  - pct_minority   : Census 2021 TS021, % non-White
Clusters = paper's KMeans (k=5, reference-aligned) from England_ML.csv; linked via home MSOA.
  Analysis A: boxplots of each dimension by lifestyle cluster (+ Kruskal-Wallis)
  Analysis B: cluster stability (Feb 2020 -> Mar 2021) by tercile of each dimension (+ chi-square)
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

ML = "/workplace/Mob/ML"; FA = "/workplace/Mob/fig_output/final_analysis_data.csv"
CEN = "/workplace/Mob/census_2021"; OUT = "/workplace/Mob/ses_analysis"; os.makedirs(OUT, exist_ok=True)
BASE = ['workplace_frequency','t1_frequency','t3_frequency','t5_frequency','t6_frequency','t8_frequency','t9_frequency']
REF = np.array([[0.4,1.2,0.4,0.9,1.0,1.9,0.3],[0.6,4.7,1.5,3.0,3.3,6.5,0.8],
                [0.6,2.6,0.9,1.8,2.0,3.8,0.6],[0.8,1.2,0.4,0.9,0.9,1.9,0.2],
                [0.7,8.9,2.9,5.4,6.2,12.1,1.2]])


def get_clusters():
    df = pd.read_csv(f"{ML}/England_ML.csv")
    feb = df[df.month==202002].copy(); mar = df[df.month==202003].copy(); mar21 = df[df.month==202103].copy()
    for c in BASE:
        feb[f"{c}_z"]=stats.zscore(feb[c]); mar[f"{c}_z"]=stats.zscore(mar[c]); mar21[f"{c}_z"]=stats.zscore(mar21[c])
    mar21.columns = ['device_aid'] + [c+'_202103' for c in mar21.columns[1:]]
    tem = feb.merge(mar, on='device_aid', how='inner', suffixes=('_202002','_202003'))
    ds = tem.merge(mar21, on='device_aid', how='inner', suffixes=('','_202103'))
    z = [f"{x}_z" for x in BASE]
    X = [ds[[i+s for i in z]].copy() for s in ['_202002','_202003','_202103']]
    g = ['workplace','a1','a3','a5','a6','a8','a9']
    for x in X: x.columns = g
    lab = KMeans(n_clusters=5, random_state=42, n_init=10).fit(pd.concat(X)).labels_
    n2 = len(X[0])
    ds['cluster_202002']=lab[:n2]; ds['cluster_202003']=lab[n2:2*n2]; ds['cluster_202103']=lab[2*n2:]
    mine = ds.groupby('cluster_202002')[[f'{c}_202002' for c in BASE]].mean().reindex(range(5)).values
    r,c = linear_sum_assignment(cdist(mine, REF)); remap = {int(a):int(b) for a,b in zip(r,c)}
    for col in ['cluster_202002','cluster_202003','cluster_202103']: ds[col] = ds[col].map(remap)
    print(f"[INFO] clustered devices: {len(ds):,} | remap {remap}")
    return ds[['device_aid','cluster_202002','cluster_202003','cluster_202103']]


def _read_msoa(tbl):
    z = zipfile.ZipFile(f"{CEN}/{tbl}.zip")
    return pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if 'msoa' in n][0])))


def census_ses():
    # occupation: NS-SeC managerial & professional (L1-L6)
    occ = _read_msoa("ts062")
    tot = occ[[c for c in occ.columns if 'Total: All usual residents' in c][0]]
    hi = occ[[c for c in occ.columns if 'L1, L2 and L3' in c][0]]; lo = occ[[c for c in occ.columns if 'L4, L5 and L6' in c][0]]
    occ['pct_manag_prof'] = (hi+lo)/tot*100
    out = occ[['geography code','pct_manag_prof']].rename(columns={'geography code':'msoa'})
    # age: median age from 5-year bands
    age = _read_msoa("ts007a")
    band_cols = [c for c in age.columns if c.startswith('Age: Aged')]
    lowers = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
    widths = [5]*17 + [10]
    counts = age[band_cols].values.astype(float); total = counts.sum(axis=1)
    cum = np.cumsum(counts, axis=1); half = total/2
    med = np.empty(len(age))
    for i in range(len(age)):
        b = np.searchsorted(cum[i], half[i])
        F = cum[i, b-1] if b > 0 else 0.0
        f = counts[i, b] if counts[i, b] > 0 else 1.0
        med[i] = lowers[b] + (half[i]-F)/f * widths[b]
    age['median_age'] = med
    out = out.merge(age[['geography code','median_age']].rename(columns={'geography code':'msoa'}), on='msoa', how='outer')
    # ethnicity: % non-White
    eth = _read_msoa("ts021")
    etot = eth[[c for c in eth.columns if c.endswith('Total: All usual residents')][0]]
    white = eth['Ethnic group: White']
    eth['pct_minority'] = (etot-white)/etot*100
    out = out.merge(eth[['geography code','pct_minority']].rename(columns={'geography code':'msoa'}), on='msoa', how='outer')
    return out


# ---------------- join (everything at MSOA) ----------------
clusters = get_clusters()
geo = pd.read_csv(FA, usecols=['device_aid','home_2021_msoa_code','income_2018']).rename(columns={'home_2021_msoa_code':'msoa'})
geo['income_2018'] = pd.to_numeric(geo['income_2018'], errors='coerce')
# income -> MSOA mean (uniform MSOA level; native income_2018 is ~LSOA-level)
inc_msoa = geo.groupby('msoa')['income_2018'].mean().rename('income_msoa')
d = clusters.merge(geo[['device_aid','msoa']], on='device_aid', how='left')
matched = d['msoa'].notna()
print(f"[INFO] clustered with home MSOA: {matched.sum():,}/{len(d):,} ({matched.mean():.1%})")
d = d[matched].merge(inc_msoa, on='msoa', how='left').merge(census_ses(), on='msoa', how='left')

DIMS = [('income_msoa', 'Neighbourhood income (£)'),
        ('pct_manag_prof', '% managerial & professional'),
        ('median_age', 'Median age (years)'),
        ('pct_minority', '% ethnic minority')]
sns.set_context("paper"); sns.set_style("ticks")

# ---------------- Analysis A ----------------
fig, axs = plt.subplots(1, 4, figsize=(17, 4.0))
print("\n[A] mean by lifestyle cluster (Feb 2020):")
for ax, (var, lab) in zip(axs, DIMS):
    sub = d.dropna(subset=[var, 'cluster_202002'])
    sns.boxplot(data=sub, x='cluster_202002', y=var, ax=ax, showfliers=False, color='#5b9bd5')
    H, p = stats.kruskal(*[g[var].values for _, g in sub.groupby('cluster_202002')])
    ax.set_title(f"{lab}\nKW H={H:.0f}, p={p:.1e}", fontsize=9)
    ax.set_xlabel('Lifestyle cluster (Feb 2020)'); ax.set_ylabel(lab)
print(d.groupby('cluster_202002')[[v for v, _ in DIMS]].mean().round(1).to_string())
sns.despine(); plt.tight_layout()
plt.savefig(f"{OUT}/A_ses_by_cluster.png", dpi=300, bbox_inches='tight'); plt.close()
print(f"[FIG] {OUT}/A_ses_by_cluster.png")

# ---------------- Analysis B ----------------
# "Stability" = a device is in the SAME lifestyle cluster at the later month as in
# its baseline month Feb 2020 (202002, pre-COVID). Two transitions:
d['stable_lockdown'] = (d['cluster_202002'] == d['cluster_202003']).astype(int)  # Feb 2020 -> Mar 2020 (first lockdown)
d['stable_covid']    = (d['cluster_202002'] == d['cluster_202103']).astype(int)  # Feb 2020 -> Mar 2021 (1 year into COVID)
TRANS = [('stable_lockdown', 'Feb 2020 → Mar 2020  (first lockdown)', '#9ecae1'),
         ('stable_covid',    'Feb 2020 → Mar 2021  (one year into COVID)', '#08519c')]
fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
print("\n[B] stability (% still in the Feb-2020 lifestyle cluster) by tercile:")
for ax, (var, lab) in zip(axs, DIMS):
    sub = d.dropna(subset=[var]).copy()
    sub['grp'] = pd.qcut(sub[var].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
    rate = sub.groupby('grp')[[t[0] for t in TRANS]].mean() * 100
    rate.columns = [t[1] for t in TRANS]
    rate.plot(kind='bar', ax=ax, color=[t[2] for t in TRANS], width=0.8,
              edgecolor='black', linewidth=0.4, legend=False)
    p_cov = stats.chi2_contingency(pd.crosstab(sub['grp'], sub['stable_covid']))[1]
    ax.set_title(f"{lab}\n(COVID-transition $\\chi^2$ p={p_cov:.0e})", fontsize=9)
    ax.set_xlabel(''); ax.set_ylabel('Still in same lifestyle cluster (%)'); ax.set_ylim(0, 70)
    plt.setp(ax.get_xticklabels(), rotation=0)
    r_lock = sub.groupby('grp')['stable_lockdown'].mean() * 100
    r_cov = sub.groupby('grp')['stable_covid'].mean() * 100
    print(f"  {var:14} lockdown[" + " ".join(f"{k}={v:.1f}" for k, v in r_lock.items()) +
          "]  covid[" + " ".join(f"{k}={v:.1f}" for k, v in r_cov.items()) + "]")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=10)
fig.suptitle("Lifestyle-cluster stability by neighbourhood SES tercile\n"
             "(% of individuals still in their Feb-2020 / pre-COVID lifestyle cluster)",
             y=1.13, fontsize=11)
sns.despine(); plt.tight_layout()
plt.savefig(f"{OUT}/B_stability_by_ses.png", dpi=300, bbox_inches='tight'); plt.close()
print(f"[FIG] {OUT}/B_stability_by_ses.png")

# ---------------- Export the minimal figure data (aggregate only, no device IDs) ----------------
# A (boxplot): per cluster the 5-number summary + mean & 95% CI of the mean.
def _boxstats(s):
    s = s.dropna().values
    q1, med, q3 = np.percentile(s, [25, 50, 75]); iqr = q3 - q1
    lo = s[s >= q1 - 1.5 * iqr].min(); hi = s[s <= q3 + 1.5 * iqr].max()
    m = s.mean(); se = s.std(ddof=1) / np.sqrt(len(s))
    return dict(n=len(s), q1=q1, median=med, q3=q3, whisker_low=lo, whisker_high=hi,
                mean=m, mean_ci95_low=m - 1.96 * se, mean_ci95_high=m + 1.96 * se)

rowsA = []
for var, lab in DIMS:
    for cl, g in d.dropna(subset=[var, 'cluster_202002']).groupby('cluster_202002'):
        rowsA.append({'dimension': lab, 'cluster': int(cl), **_boxstats(g[var])})
pd.DataFrame(rowsA).round(3).to_csv(f"{OUT}/A_ses_by_cluster_data.csv", index=False)
print(f"[DATA] {OUT}/A_ses_by_cluster_data.csv")

# B (bar): per tercile x transition the stability % + 95% CI of the proportion (Wald).
rowsB = []
for var, lab in DIMS:
    sub = d.dropna(subset=[var]).copy()
    sub['grp'] = pd.qcut(sub[var].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
    for grp, g in sub.groupby('grp'):
        for tcol, tlab in [('stable_lockdown', 'Feb2020->Mar2020 (first lockdown)'),
                           ('stable_covid', 'Feb2020->Mar2021 (one year into COVID)')]:
            pr = g[tcol].mean(); n = len(g); se = np.sqrt(pr * (1 - pr) / n)
            rowsB.append({'dimension': lab, 'ses_tercile': str(grp), 'transition': tlab, 'n': n,
                          'stability_pct': round(pr * 100, 2),
                          'ci95_low': round((pr - 1.96 * se) * 100, 2),
                          'ci95_high': round((pr + 1.96 * se) * 100, 2)})
pd.DataFrame(rowsB).to_csv(f"{OUT}/B_stability_by_ses_data.csv", index=False)
print(f"[DATA] {OUT}/B_stability_by_ses_data.csv")

# ---------------- Full per-row data behind each figure (no device IDs / MSOA codes) ----------------
full = d.copy()
for var, _ in DIMS:
    full[f'{var}_tercile'] = pd.qcut(full[var].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
full[['cluster_202002'] + [v for v, _ in DIMS]].round(3).to_csv(
    f"{OUT}/A_ses_by_cluster_FULLDATA.csv", index=False)                       # boxplot point cloud
full[[v for v, _ in DIMS] + [f'{v}_tercile' for v, _ in DIMS] + ['stable_lockdown', 'stable_covid']].round(3).to_csv(
    f"{OUT}/B_stability_by_ses_FULLDATA.csv", index=False)                     # bar inputs (tercile + stability)
print(f"[DATA] full-data A ({len(full):,} rows) + B written")

# slim all-months table (no device IDs / MSOA codes) — drives the per-month figures
SLIM = ['cluster_202002', 'cluster_202003', 'cluster_202103', 'income_msoa',
        'pct_manag_prof', 'median_age', 'pct_minority', 'stable_lockdown', 'stable_covid']
d[SLIM].round(3).to_csv(f"{OUT}/clusters_ses_allmonths.csv", index=False)
print(f"[DATA] {OUT}/clusters_ses_allmonths.csv ({len(d):,} rows)")
print(f"\n[DONE] {OUT}")
