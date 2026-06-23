# UK Mobility Analysis

A research pipeline that turns raw device-level location pings into **individual mobility
behaviour profiles** for the UK (England), and analyses how people redistributed their time
across **home / workplace / amenities** across 2020–2022 (a COVID-era before/after comparison).

The pipeline ingests daily location data, identifies each device's home and workplace from
stay durations, detects amenity (POI) visits, weights visit time/frequency by POI type, and
finally clusters individuals into mobility typologies with cross-year transition analysis.

> ⚠️ **Data privacy.** This repository contains **code only**. The underlying data is
> device-level human-location data and is **not** included. Notebook outputs have been
> stripped and example device identifiers redacted so that no personal/location records are
> committed. Do not commit raw data, intermediate CSV/Parquet, or `.RData` files (see
> `.gitignore`).

---

## Pipeline overview

```
raw daily .RData location pings
        │
        ▼  s0  ingest & filter by hours-present
   processed daily CSVs  ──────────────────────────────┐
        │                                               │
        ▼  s1  home / workplace identification          │
   per-device home & workplace + visit classification   │
        │                                               │
        ▼  s2  amenity / POI stop detection             │
   amenity visits                                        │
        │                                               │
        ▼  s3  weight by POI typology                   │
   weighted hours & frequency per amenity type          │
        │                                               │
        ▼  s4  individual profiles → clustering         │
   mobility typologies + 2020↔2021/2022 transitions ◄───┘
```

### Stage → file map

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| **s0 · Ingest & filter** | `src/s0_ingest/filter_raw_pings.py`, `src/s0_ingest/enrich_daily_records.py` | Read raw daily `.RData` pings (UK 2020/2021/2022), keep devices with enough hours present, derive per-location stay durations and daily features. |
| **s1 · Home / workplace ID** | `src/s1_home_workplace/home_workplace_pipeline.py` | Identify each device's home & workplace from stay duration, aggregate to yearly home/work locations, filter to devices valid across years, and classify every visit as `home` / `workplace` / `amenity`. |
| **s2 · Amenity detection** | `src/s2_amenity/detect_amenity_stops.py`, `src/s2_amenity/select_by_device.py` | Detect POI/amenity stops from movement traces and collect per-device amenity records (filtered to valid device IDs). |
| **s3 · Weighted hours & frequency** | `src/s3_weighted_hours/weighted_hours.py` (+ `weighted_hours.sh`, `weighted_hours_backup.py`) | Join amenity visits to a POI typology (eating/drinking, attractions, health, public infrastructure, retail, transport) and compute weighted hours and visit frequency per type. Run via `--year {2021,2022,2020_for_2021,2020_for_2022}`. |
| **s4 · Profiles, clustering & viz** | `src/s4_clustering/cluster_profiles.py`, `src/s4_clustering/cluster_profiles_individual.py` | Build individual mobility profiles, standardize/PCA, KMeans clustering (with Hungarian-algorithm label alignment across years), cross-year transition matrices, and ternary (home/work/amenity) distribution plots. |
| **utils** | `src/utils/sample_devices.py` | Ad-hoc helper to sample a subset of devices for quick testing. |

### Notebooks (`notebooks/`, outputs stripped)

- `s0_enrich_daily_records.ipynb` — prototype of the ingest / enrichment script.
- `s1_home_workplace_explore.ipynb`, `s1_home_workplace_dev.ipynb` — home/work identification development.
- `s2_amenity_detection.ipynb`, `s2_amenity_detection_v2.ipynb` — amenity-detection prototypes.
- `s3_frequency_and_time.ipynb` — frequency & time / weighted-hours exploration.
- `s4_clustering.ipynb`, `s4_clustering_individual.ipynb`, `s4_ternary_plots.ipynb` — clustering and ternary-plot exploration.
- `aux_safety_perception.ipynb` — auxiliary street-view safety-perception check.

---

## Repository layout

```
.
├── src/
│   ├── s0_ingest/
│   │   ├── filter_raw_pings.py            # filter raw RData by hours-present
│   │   └── enrich_daily_records.py        # enrich daily records → processed CSV
│   ├── s1_home_workplace/
│   │   └── home_workplace_pipeline.py     # home/workplace identification pipeline
│   ├── s2_amenity/
│   │   ├── detect_amenity_stops.py        # amenity stop detection
│   │   └── select_by_device.py            # collect per-device amenity records
│   ├── s3_weighted_hours/
│   │   ├── weighted_hours.py              # POI-weighted hours & frequency (per year)
│   │   ├── weighted_hours.sh              # driver for the above
│   │   └── weighted_hours_backup.py       # backup variant
│   ├── s4_clustering/
│   │   ├── cluster_profiles.py            # clustering + transition + ternary
│   │   └── cluster_profiles_individual.py # individual-level variant
│   └── utils/
│       └── sample_devices.py              # ad-hoc device sampling
├── notebooks/                             # exploratory / figure notebooks (outputs stripped)
├── figures/                              # example output figures (.png / .pdf)
├── data/
│   └── England_individual_1127_columns.csv  # column schema reference (no data rows)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pyreadr` is required to read the R `.RData`/`.R` source files; `python-ternary` is imported as
`ternary` for the ternary plots.

## Usage notes

- **Paths are hardcoded** to the original compute environment (e.g. `/nas/houce/UK_mobility/...`).
  Update the path constants near the top of each script (`BASE_DIR`, `DATA_PATH`,
  `--base_path`, etc.) to point at your own data before running.
- Run scripts from the repository root, e.g.:
  ```bash
  python src/s0_ingest/filter_raw_pings.py
  python src/s0_ingest/enrich_daily_records.py
  python src/s1_home_workplace/home_workplace_pipeline.py
  python src/s2_amenity/detect_amenity_stops.py
  python src/s2_amenity/select_by_device.py
  python src/s3_weighted_hours/weighted_hours.py --year 2022
  python src/s4_clustering/cluster_profiles.py
  ```

## POI typology

| Code | Category |
|------|----------|
| `t1` | Eating & drinking |
| `t3` | Attractions & entertainment |
| `t5` | Health |
| `t6` | Public infrastructure |
| `t8` | Retail |
| `t9` | Transport |

---

## File renames (old → new)

The repository was reorganised from a flat layout into pipeline-stage folders. Mapping:

| Old | New |
|-----|-----|
| `filter_ori_files.py` | `src/s0_ingest/filter_raw_pings.py` |
| `data_process_20251212.py` | `src/s0_ingest/enrich_daily_records.py` |
| `A_0_Shizhen_process/total_pipeline.py` | `src/s1_home_workplace/home_workplace_pipeline.py` |
| `amenity_detection.py` | `src/s2_amenity/detect_amenity_stops.py` |
| `ML.py` | `src/s2_amenity/select_by_device.py` |
| `A_0_Shizhen_process/A_4_get_weighted_hours.py` | `src/s3_weighted_hours/weighted_hours.py` |
| `A_0_Shizhen_process/A_4_get_weighted_hours.sh` | `src/s3_weighted_hours/weighted_hours.sh` |
| `A_0_Shizhen_process/A_4_backup.py` | `src/s3_weighted_hours/weighted_hours_backup.py` |
| `ML_20251214.py` | `src/s4_clustering/cluster_profiles.py` |
| `ML_20251214_individual.py` | `src/s4_clustering/cluster_profiles_individual.py` |
| `tem_20251220.py` | `src/utils/sample_devices.py` |
| `data_process_20251212.ipynb` | `notebooks/s0_enrich_daily_records.ipynb` |
| `A_0_Shizhen_process/A_0_Shizhen_process.ipynb` | `notebooks/s1_home_workplace_explore.ipynb` |
| `A_0_Shizhen_process/A_1_Shizhen_process.ipynb` | `notebooks/s1_home_workplace_dev.ipynb` |
| `A_0_Shizhen_process/A_3_frequency_and_time.ipynb` | `notebooks/s3_frequency_and_time.ipynb` |
| `amenity_detection.ipynb` | `notebooks/s2_amenity_detection.ipynb` |
| `amenity_detection2.ipynb` | `notebooks/s2_amenity_detection_v2.ipynb` |
| `ML.ipynb` | `notebooks/s4_clustering.ipynb` |
| `ML_individual.ipynb` | `notebooks/s4_clustering_individual.ipynb` |
| `triangle_draw.ipynb` | `notebooks/s4_ternary_plots.ipynb` |
| `safety_perception_check.ipynb` | `notebooks/aux_safety_perception.ipynb` |

---

*Code is shared for transparency/reproducibility of methodology. The mobility and POI datasets
are not distributed with this repository.*
