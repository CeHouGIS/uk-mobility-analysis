#!/usr/bin/env python
"""
Per-month version of Analysis A: SES (income / occupation / age / ethnicity) by lifestyle
cluster, using each month's cluster assignment (Feb 2020, Mar 2020, Mar 2021).
Reads the regenerated device table; writes one figure + one data table per month.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = "/workplace/Mob/ses_analysis"
d = pd.read_csv(f"{OUT}/clusters_ses_allmonths.csv")
DIMS = [('income_msoa', 'Neighbourhood income (£)'), ('pct_manag_prof', '% managerial & professional'),
        ('median_age', 'Median age (years)'), ('pct_minority', '% ethnic minority')]
MONTHS = [('202002', 'Feb 2020 (pre-COVID)'), ('202003', 'Mar 2020 (first lockdown)'),
          ('202103', 'Mar 2021 (one year into COVID)')]
sns.set_context("paper"); sns.set_style("ticks")


def box(s):
    s = s.dropna().values
    q1, med, q3 = np.percentile(s, [25, 50, 75]); iqr = q3 - q1
    lo = s[s >= q1 - 1.5 * iqr].min(); hi = s[s <= q3 + 1.5 * iqr].max()
    m = s.mean(); se = s.std(ddof=1) / np.sqrt(len(s))
    return dict(n=len(s), q1=q1, median=med, q3=q3, whisker_low=lo, whisker_high=hi,
                mean=m, mean_ci95_low=m - 1.96 * se, mean_ci95_high=m + 1.96 * se)


for m, mlab in MONTHS:
    cl = f'cluster_{m}'
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    rows = []
    for ax, (var, lab) in zip(axs, DIMS):
        sub = d.dropna(subset=[var, cl])
        sns.boxplot(data=sub, x=cl, y=var, ax=ax, showfliers=False, color='#5b9bd5')
        H, p = stats.kruskal(*[g[var].values for _, g in sub.groupby(cl)])
        ax.set_title(f"{lab}\nKW H={H:.0f}, p={p:.1e}", fontsize=9)
        ax.set_xlabel('Lifestyle cluster'); ax.set_ylabel(lab)
        for c, g in sub.groupby(cl):
            rows.append({'month': mlab, 'dimension': lab, 'cluster': int(c), **box(g[var])})
    fig.suptitle(f"SES by lifestyle cluster — {mlab}", y=1.04, fontsize=12)
    sns.despine(); plt.tight_layout()
    fig.savefig(f"{OUT}/A_ses_by_cluster_{m}.png", dpi=300, bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(rows).round(3).to_csv(f"{OUT}/A_ses_by_cluster_{m}_data.csv", index=False)
    print(f"[FIG] A_ses_by_cluster_{m}.png   [DATA] A_ses_by_cluster_{m}_data.csv  ({m})")
print("[DONE]")
