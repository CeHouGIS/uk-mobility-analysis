# Socioeconomic heterogeneity of lifestyle clusters & behavioral inertia

Addresses reviewer concerns on (a) the socioeconomic dimension of behavioral inertia and
(b) analytical independence of K-means vs the fixed-effects models — the SES variables are
**external** (UK Census 2021 + ONS income), so this is independent validation, not the same
data re-confirming itself.

`ses_analysis.py` pipeline:
1. Reproduce the paper's KMeans clusters (k=5, reference-aligned) from `England_ML.csv`.
2. Link `device → home MSOA` via `final_analysis_data.csv` (also gives `income_2018`).
3. Build four **neighbourhood (MSOA-level)** SES/demographic dimensions, join on home MSOA.

| Dimension | Variable | Source | Level |
|-----------|----------|--------|-------|
| Income | `income_msoa` | ONS income (`income_2018`) averaged to MSOA | MSOA* |
| Occupational | `pct_manag_prof` (% NS-SeC L1–L6) | Census 2021 **TS062** | MSOA |
| Demographic – age | `median_age` (from 5-yr bands) | Census 2021 **TS007a** | MSOA |
| Demographic – ethnicity | `pct_minority` (% non-White) | Census 2021 **TS021** | MSOA |

\* `income_2018` is natively ~LSOA-level; aggregated to the MSOA mean so all four dimensions
are uniformly MSOA. Home MSOA codes match the 2021 census codes at ~98%.

## Analysis A — SES by lifestyle cluster (`A_ses_by_cluster.png`)
Boxplots of each dimension by cluster (+ Kruskal-Wallis). Lifestyle clusters **are**
socioeconomically stratified; ethnicity and age separate them most strongly — the
high-activity cluster is younger, more ethnically diverse and higher-income.

## Analysis B — inertia by SES tercile (`B_stability_by_ses.png`)
"Stability" = a device is in the **same lifestyle cluster** at a later month as in its
**baseline month Feb 2020 (pre-COVID)**. Two transitions are shown per SES tercile:
**Feb 2020 → Mar 2020** (first lockdown, ~60–62% stable) and **Feb 2020 → Mar 2021**
(one year into COVID, ~44–49% stable — stability decays with the longer gap).
Directions below refer to the one-year (COVID) transition.

| Dimension | Direction | Low → High stability |
|-----------|-----------|----------------------|
| Income | higher → less inertia | 47.7 → 44.1% |
| % managerial/professional | higher → less inertia | 47.7 → 45.0% |
| Median age | older → **more** inertia | 45.3 → 47.9% |
| % ethnic minority | more diverse → less inertia | 49.1 → 44.0% |

Coherent story: affluent / younger / more-diverse neighbourhoods changed behaviour more
during COVID (lower inertia); lower-SES / older / more-homogeneous areas were more "stuck".

## Caveats (for the rebuttal)
- **Ecological**: SES is neighbourhood (MSOA) level, not individual — report as "neighbourhood SES".
- **Coverage**: 50,648 / 105,347 clustered devices (48%) have home geography; matched vs unmatched
  cluster mix is similar (no strong selection) but should be reported.
- **Pseudo-replication**: many devices share an MSOA, so the very small p-values reflect N +
  non-independence, not large effects (effects are ≈3–5 pp). A final version should aggregate to
  MSOA or use a mixed model (MSOA random effect).

## Per-month figures & data exports
- `A_ses_by_cluster_{202002,202003,202103}.png` — Analysis A repeated with **each month's**
  cluster (Feb 2020 / Mar 2020 / Mar 2021). Generator: `per_month_ses.py`.
- Minimal figure data (with 95% CI): `A_ses_by_cluster_data.csv` (per cluster: 5-number summary
  + mean & 95% CI), `B_stability_by_ses_data.csv` (per tercile×transition: stability % + 95% CI),
  and per-month `A_ses_by_cluster_<month>_data.csv`.
- Full per-row plotting data (no device IDs / MSOA codes): `A_ses_by_cluster_FULLDATA.csv`
  (cluster + 4 SES values per device) and `B_stability_by_ses_FULLDATA.csv` (SES + terciles +
  stability flags). `clusters_ses_allmonths.csv` = slim per-device table (all 3 months' clusters
  + SES + stability), also ID-free.

Data note: only ID-free aggregate/row data is committed. The device-ID-keyed table and the Census
bulk files are **not** committed. Census tables: ONS Census 2021 via Nomis bulk
(`census2021-ts062 / ts007a / ts021`).
