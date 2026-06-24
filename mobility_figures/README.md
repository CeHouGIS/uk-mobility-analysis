# Regenerated mobility figures

Run any script with `/opt/conda/bin/python <script>.py`.

- `redraw_from_final_analysis.py` → **`faithful_latest/`** — the real deal: original
  `cluster_profiles.py` figures redrawn from `fig_output/final_analysis_data.csv`
  (the wide table) **using its existing `cluster_2020`/`cluster_2021` labels** (no
  re-clustering), so cluster contents/numbering match the original `fig_output`
  reference figures. Ternary uses LAD-normalised days.
  Files: `feature_means_original_values_{2020,2021}.png`, `pca_clusters_{2020,2021}.png`,
  `ternary_distribution_{2020,2021}.png`, `transition_matrix_{counts,percent}.png`.
- `make_figures.py` — quick first pass on the long table (root folder + `adapted_clustering/`).
- `make_figures_notebook_style.py` — **prettier** set in `notebook_style/`, styling
  ported from `notebooks/s4_ternary_plots.ipynb` and `notebooks/s4_clustering.ipynb`.

## `notebook_style/` (recommended — ported from the plotting notebooks)
- `ternary_heatmap_2020XX.png` (×4) — ternary density heatmap, notebook styling
  (heavy boundary, black+blue dual gridlines, fontsize-18 title), Home/Work/Amenities
  weighted-hours shares per period.
- `ternary_scatter_cluster_202003.png` — ternary scatter coloured by cluster with a
  colorbar (port of `s4_ternary_plots.ipynb` cell 7).
- `cluster_means_standardized_202003.png` — `RdBu_r` centered cluster-mean heatmap
  (port of `s4_clustering.ipynb` cell 13).
- `elbow_202003.png` — SSE elbow plot (port of `s4_clustering.ipynb` cell 6).

> Note: the notebooks originally read the WIDE per-individual table
> (`home_days_2020`, `cluster_2020`, LAD codes) which isn't available here, so these
> use raw weighted-hours shares from the long table instead of LAD-normalised days.

## About the input data
`England_ML.csv` is a **long-format** table: one row per `(device_aid, month)`,
369,090 rows / 133,733 devices, columns = weighted hours + visit frequency for
home, workplace and 6 amenity types (t1,t3,t5,t6,t8,t9). Four periods:

| month | label | rows |
|-------|-------|------|
| 202002 | Feb 2020 (pre-COVID) | 127,172 |
| 202003 | Mar 2020 (1st lockdown) | 115,921 |
| 202103 | Mar 2021 | 121,344 |
| 202203 | Mar 2022 | 4,653 (sparse) |

> NOTE: this file does **not** contain the wide, year-suffixed, LAD-coded
> per-individual columns (`home_hours_2020`, `days_t1_2020`, `home_2020_lad_code`,
> `home_same`, …) that the repo script `src/s4_clustering/cluster_profiles.py`
> clusters on. So that script cannot be run on this file as-is.

## FAITHFUL figures (directly supported by this data)
- `ternary_2020XX.png` (×4) — Home / Workplace / Amenities time-share ternary
  density per period. Same idea as the repo's `visualize_ternary_distribution`,
  but on **raw weighted-hours shares** (no LAD normalization).
- `weighted_hours_heatmap.png` — mean weighted hours/day, place × period.
- `frequency_heatmap.png` — mean visit frequency, place × period.
- `mean_weighted_hours_by_period.png` — grouped-bar version of the above.
- `weighted_hours_boxplot_by_period.png` — distribution of home/workplace/amenity
  weighted hours per period.

## ADAPTED clustering (`adapted_clustering/`) — different methodology
Analog of `cluster_profiles.py`'s cluster figures, but clustered **directly on
the 8 weighted-hours features** (StandardScaler + KMeans, k=7), comparing
**Mar 2020 vs Mar 2021** for the 108,518 devices present in both. This is NOT the
repo method (no LAD regional-ratio normalization, periods instead of calendar
years), so treat these as exploratory.
- `cluster_feature_means_202003.png` — cluster mean weighted hours (Mar 2020).
- `pca_clusters_202003.png` — PCA scatter of clusters (Mar 2020).
- `transition_matrix_counts.png` / `transition_matrix_percent.png` — cluster
  transition Mar 2020 → Mar 2021 (Hungarian-aligned labels). Overall change ≈ 49%.

## To reproduce the ORIGINAL cluster figures
Provide the wide per-individual profile table (the real `England_ML.csv` with
year-suffixed + LAD columns, schema in `data/England_individual_1127_columns.csv`)
and run `src/s4_clustering/cluster_profiles.py` with `DATA_PATH` pointed at it.
