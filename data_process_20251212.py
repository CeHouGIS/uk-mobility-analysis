# python /code/Mob/data_process_20251212.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import pandas as pd
import pyreadr
from glob import glob


def load_rdata(path: str | Path) -> pd.DataFrame:
    """读取 RData 文件并返回其中首个数据框。"""
    result: Dict[str, pd.DataFrame] = pyreadr.read_r(str(path))
    df = next(iter(result.values())).reset_index(drop=True)
    if "device_aid" not in df.columns:
        raise KeyError(f"'device_aid' not found in {path}")
    return df


def extract_day_tag(path: str | Path) -> str:
    """从文件名中提取日期标签，如 UK_2020_0201.RData -> 20200201。"""
    if ('2020 Feb' in path) or ('2022 March' in path):
        match = re.search(r"UK_(\d{4})_(\d{4})\.RData$", Path(path).name)
    elif ('2020 March' in path) or ('2021 March' in path):
        match = re.search(r"uk_(\d{4})_(\d{4})\.R$", Path(path).name)

    if not match:
        raise ValueError(f"Unexpected filename format: {path}")
    year, md = match.groups()
    return f"{year}{md}"


def enrich_daily_records(df: pd.DataFrame, day_tag: str) -> pd.DataFrame:
    """新增地点/小时特征并聚合。"""
    df = df.copy()
    df["day"] = day_tag
    df["location"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    if "hours" not in df.columns:
        df["showed_exact_hours"] = df["hour"].str.split().str[-1].astype(int)
        df.drop(columns=["hour"], inplace=True)
    else:
        df["showed_exact_hours"] = df["hours"].astype(int)

    df["hour_count"] = df.groupby(["device_aid", "showed_exact_hours"])["showed_exact_hours"].transform("count")
    df["place_count"] = (
        df.groupby(["device_aid", "location", "showed_exact_hours"])["showed_exact_hours"].transform("count")
    )
    df["single_location_hour"] = df["place_count"] / df["hour_count"]
    df["showed_hours_today"] = df.groupby(["device_aid", "day"])["showed_exact_hours"].transform("nunique")
    df["duration_location_hour"] = df.groupby(["device_aid", "location"])["single_location_hour"].transform("sum")

    return (
        df.drop(columns=["showed_exact_hours", "place_count", "hour_count", "single_location_hour"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    base_dir = Path("/nas/houce/UK_mobility")
    # keep_path = base_dir / "device_aid_keep_England.Rdata"
    # print(f"[INFO] Loading keep list: {keep_path}")
    # df_keep = load_rdata(keep_path).rename(columns={"device_aid_keep": "device_aid"})
    # print(f"[INFO] Keep list loaded: {len(df_keep):,} rows")

    daily_files = glob(str(base_dir / "2022 March" / "*"))
    print(f"[INFO] Found {len(daily_files)} daily files to process")
    outputs_dir = base_dir / "processed_data"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for idx, file_path in tqdm(enumerate(sorted(daily_files), 1), total=len(daily_files)):
        out_path = outputs_dir / f"{Path(file_path).stem}_processed.csv"
        if out_path.exists():
            print(f"\n[INFO] ({idx}/{len(daily_files)}) Skipping {file_path}, output exists.")
            continue
        print(f"\n[INFO] ({idx}/{len(daily_files)}) Processing {file_path}")
        day_tag = extract_day_tag(file_path)
        df_raw = load_rdata(file_path)
        print(f"        Raw rows: {len(df_raw):,}")
        df_daily = enrich_daily_records(df_raw, day_tag)
        print(f"        Enriched rows: {len(df_daily):,}")
        # merged = df_keep.merge(df_daily, on="device_aid", how="inner")
        # print(f"        Merged rows: {len(merged):,}")
        df_daily.to_csv(out_path, index=False)
        print(f"        Saved output to {out_path}")

    print("\n[INFO] Processing complete.")