# Socioeconomic heterogeneity of lifestyle clusters & behavioral inertia

Addresses reviewer concerns on (a) the socioeconomic dimension of behavioral inertia and
(b) analytical independence of K-means vs the fixed-effects models — the SES variables are
**external** (UK Census 2021), so this is independent validation, not the same data re-confirming itself.

`ses_analysis.py` — pipeline (run with the data files in place):
1. Reproduce the paper's KMeans clusters (k=5, reference-aligned) from `England_ML.csv`.
2. Link `device → home MSOA` via `final_analysis_data.csv` (also gives `income_2018`).
3. Build MSOA-level SES from **2021 Census** (Nomis bulk): `pct_degree` (TS067, % Level 4+),
   `pct_manag_prof` (TS062 NS-SeC, % managerial & professional). Home MSOA codes match the
   2021 census codes at ~98%.

## Analysis A — SES by lifestyle cluster (`A_ses_by_cluster.png`)
Boxplots of income / % degree / % managerial-professional by cluster (+ Kruskal-Wallis).
Lifestyle clusters **are** socioeconomically stratified: higher-activity clusters sit in
higher-income, higher-education neighbourhoods. Effect sizes are modest (≈£3.5k median income,
≈5pp degree).

## Analysis B — inertia by SES group (`B_stability_by_ses.png`)
Share staying in the same lifestyle cluster, Feb 2020 → Mar 2020 and Feb 2020 → Mar 2021,
by neighbourhood-income tercile. Higher-income areas are **less** stable over COVID
(47.9% Low → 44.1% High for the 2021 transition; χ² p≈2e-13) — i.e. behavioral inertia is
stronger in lower-SES neighbourhoods.

## Caveats (for the rebuttal)
- **Ecological**: SES is neighbourhood (MSOA) level, not individual — report as "neighbourhood SES".
- **Coverage**: 50,648 / 105,347 clustered devices (48%) have home geography; cluster mix of the
  matched vs unmatched subsample is similar (no strong selection), but should be reported.
- **Pseudo-replication**: many devices share an MSOA, so the very small p-values reflect N +
  non-independence, not large effects. A final version should aggregate to MSOA or use a
  mixed model (MSOA random effect).

Data note: the device-level joined table (`clusters_with_ses.csv`) contains device IDs + home
MSOA and is **not** committed.

Sources: ONS Census 2021 via Nomis bulk (`census2021-ts067`, `census2021-ts062`).
