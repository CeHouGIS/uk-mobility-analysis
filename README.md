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
        ▼  (0) ingest & filter by hours-present
   processed daily CSVs  ──────────────────────────────┐
        │                                               │
        ▼  (1) home / workplace identification          │
   per-device home & workplace + visit classification   │
        │                                               │
        ▼  (2) amenity / POI stop detection             │
   amenity visits                                        │
        │                                               │
        ▼  (3) weight by POI typology                   │
   weighted hours & frequency per amenity type          │
        │                                               │
        ▼  (4) individual profiles → clustering         │
   mobility typologies + 2020↔2021/2022 transitions ◄───┘
```

### Stage → file map

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| **0. Ingest & filter** | `filter_ori_files.py`, `data_process_20251212.py` | Read raw daily `.RData` pings (UK 2020/2021/2022), keep devices with enough hours present, derive per-location stay durations and daily features. |
| **1. Home / workplace ID** | `A_0_Shizhen_process/total_pipeline.py` | Identify each device's home & workplace from stay duration, aggregate to yearly home/work locations, filter to devices valid across years, and classify every visit as `home` / `workplace` / `amenity`. |
| **2. Amenity detection** | `amenity_detection.py`, `ML.py` | Detect POI/amenity stops from movement traces and collect per-device amenity records. |
| **3. Weighted hours & frequency** | `A_0_Shizhen_process/A_4_get_weighted_hours.py` (+ `.sh`, `A_4_backup.py`) | Join amenity visits to a POI typology (eating/drinking, attractions, health, public infrastructure, retail, transport) and compute weighted hours and visit frequency per type. Run via `--year {2021,2022,2020_for_2021,2020_for_2022}`. |
| **4. Profiles, clustering & viz** | `ML_20251214.py`, `ML_20251214_individual.py`, notebooks below | Build individual mobility profiles, standardize/PCA, KMeans clustering (with Hungarian-algorithm label alignment across years), cross-year transition matrices, and ternary (home/work/amenity) distribution plots. |

### Notebooks

Exploratory / figure-generating notebooks (outputs stripped):

- `A_0_Shizhen_process/A_0_Shizhen_process.ipynb`, `A_1_Shizhen_process.ipynb`, `A_3_frequency_and_time.ipynb` — home/work + frequency/time development notebooks.
- `amenity_detection.ipynb`, `amenity_detection2.ipynb`, `data_process_20251212.ipynb` — prototypes of the ingest / amenity-detection scripts.
- `ML.ipynb`, `ML_individual.ipynb`, `triangle_draw.ipynb` — clustering and ternary-plot exploration.
- `safety_perception_check.ipynb` — auxiliary street-view safety-perception check.

---

## Repository layout

```
.
├── filter_ori_files.py              # (0) filter raw RData by hours-present
├── data_process_20251212.py         # (0) enrich daily records → processed CSV
├── amenity_detection.py             # (2) amenity stop detection
├── ML.py                            # (2) collect per-device amenity records
├── ML_20251214.py                   # (4) clustering + transition + ternary
├── ML_20251214_individual.py        # (4) individual-level variant
├── tem_20251220.py                  # ad-hoc sampling helper
├── A_0_Shizhen_process/
│   ├── total_pipeline.py            # (1) home/workplace identification pipeline
│   ├── A_4_get_weighted_hours.py    # (3) POI-weighted hours & frequency
│   ├── A_4_get_weighted_hours.sh    #     driver for the above (per year)
│   ├── A_4_backup.py                #     backup variant
│   └── *.ipynb                      # development notebooks
├── England_individual_1127_columns.csv  # column schema reference (no data rows)
├── *.ipynb                          # exploratory / figure notebooks
├── *.pdf / *.png                    # example output figures
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
- Typical order: `filter_ori_files.py` → `data_process_20251212.py` →
  `A_0_Shizhen_process/total_pipeline.py` → `amenity_detection.py` / `ML.py` →
  `A_0_Shizhen_process/A_4_get_weighted_hours.py` → `ML_20251214.py`.
- The weighted-hours step is year-parameterised:
  ```bash
  python A_0_Shizhen_process/A_4_get_weighted_hours.py --year 2022
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

*Code is shared for transparency/reproducibility of methodology. The mobility and POI datasets
are not distributed with this repository.*
