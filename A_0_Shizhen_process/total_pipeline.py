# python /code/Mob/A_0_Shizhen_process/total_pipeline.py
import pandas as pd
from glob import glob
import os
import multiprocessing
from functools import partial
from tqdm import tqdm
import gc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- Configuration & Constants ---
# 路径配置
BASE_DIR = "/nas/houce/UK_mobility"
INPUT_DATA_PATH_TEMPLATE = os.path.join(BASE_DIR, "processed_data", "*_{year}_*_processed.csv")
OUTPUT_DATA_PATH = os.path.join(BASE_DIR, "data_20251226_new") # 统一输出路径
HOME_WORK_LOC_DIR = os.path.join(OUTPUT_DATA_PATH, "home_workplace_location")

# 列名常量
DEVICE_ID_COL = 'device_aid'
DURATION_COL = 'duration_location_hour'
LAT_COL = 'latitude'
LON_COL = 'longitude'
LOCATION_COLS = [DEVICE_ID_COL, LAT_COL, LON_COL]

# 数据类型优化字典
DTYPE_DICT = {
    DEVICE_ID_COL: 'str',
    LAT_COL: 'float32',
    LON_COL: 'float32',
    DURATION_COL: 'float32',
    'showed_hours_today': 'float32',
    'home': 'int8',
    'workplace': 'int8'
}

# --- Part 1: Daily Home/Workplace Identification ---
# (保持原有的高效逻辑不变)
def identify_locations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df['home'] = 0; df['workplace'] = 0
        return df
    
    max_durations = df.groupby(DEVICE_ID_COL)[DURATION_COL].transform('max')
    df['home'] = (df[DURATION_COL] == max_durations).astype('int8')

    is_not_home = df['home'] == 0
    if is_not_home.any():
        max_work_map = df[is_not_home].groupby(DEVICE_ID_COL)[DURATION_COL].max()
        df['max_work_dur'] = df[DEVICE_ID_COL].map(max_work_map)
        df['workplace'] = (is_not_home & (df[DURATION_COL] == df['max_work_dur'])).astype('int8')
        df.drop(columns=['max_work_dur'], inplace=True)
    else:
        df['workplace'] = 0
        
    df['workplace'] = df['workplace'].fillna(0).astype('int8')
    return df

def process_file(file_path: str, save_path: str, showed_hours_threshold: int = 6):
    try:
        # 尝试使用 pyarrow 引擎加速，失败则回退
        engine = 'pyarrow' if pd.__version__ >= '1.4.0' else 'c'
        try:
            df = pd.read_csv(file_path, dtype=DTYPE_DICT, engine=engine)
        except:
            df = pd.read_csv(file_path, dtype=DTYPE_DICT)

        if 'showed_hours_today' in df.columns:
            if showed_hours_threshold is not None:
                df = df[df['showed_hours_today'] >= showed_hours_threshold]
        if 'hours' in df.columns:
            df.drop(columns=['hours'], inplace=True)
        df.drop_duplicates(inplace=True)

        if df.empty: return

        df_final = identify_locations(df)

        file_name_base = os.path.basename(file_path).replace('_processed.csv', '')
        combined_output_path = os.path.join(save_path, 'home_workplace', f'{file_name_base}_home_workplace.csv')
        os.makedirs(os.path.dirname(combined_output_path), exist_ok=True)
        df_final.to_csv(combined_output_path, index=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def run_parallel_processing(files_to_process: list, save_path: str):
    if not files_to_process: return
    os.makedirs(save_path, exist_ok=True)
    worker_func = partial(process_file, save_path=save_path)
    num_processes = max(1, min(multiprocessing.cpu_count() - 2, len(files_to_process)))
    print(f"Starting parallel processing of {len(files_to_process)} files...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(worker_func, files_to_process), total=len(files_to_process)))

# --- Part 2: Yearly Aggregation ---
# (保持原逻辑不变)
def process_yearly_data_optimized(year: int, base_path: str, home_only: bool, showed_hours_threshold: int = 6):
    print(f"\n--- Starting yearly processing for {year} ---")
    file_pattern = os.path.join(base_path, 'home_workplace', f'*_{year}*_home_workplace.csv')
    file_list = sorted(glob(file_pattern))
    if not file_list: return None

    CHUNK_SIZE = 50
    aggregated_chunks = []
    
    for i in tqdm(range(0, len(file_list), CHUNK_SIZE)):
        chunk_files = file_list[i : i + CHUNK_SIZE]
        chunk_dfs = []
        for f in chunk_files:
            try:
                temp_df = pd.read_csv(f, usecols=LOCATION_COLS + [DURATION_COL, 'showed_hours_today'], dtype=DTYPE_DICT)
                if showed_hours_threshold is not None:
                    temp_df = temp_df[temp_df['showed_hours_today'] >= showed_hours_threshold]
                chunk_dfs.append(temp_df)
            except: continue
        
        if chunk_dfs:
            chunk_concat = pd.concat(chunk_dfs, ignore_index=True)
            chunk_agg = chunk_concat.groupby(LOCATION_COLS, as_index=False)[DURATION_COL].sum()
            aggregated_chunks.append(chunk_agg)
            del chunk_concat, chunk_dfs, chunk_agg
            gc.collect()

    if not aggregated_chunks: return None

    print("Finalizing aggregation...")
    total_hours_df = pd.concat(aggregated_chunks, ignore_index=True)
    total_hours_df = total_hours_df.groupby(LOCATION_COLS, as_index=False)[DURATION_COL].sum()
    
    home_output_path = os.path.join(base_path, 'home_workplace_location', f'home_location_{year}.csv')
    idx = total_hours_df.groupby(DEVICE_ID_COL)[DURATION_COL].idxmax()
    home_locations = total_hours_df.loc[idx].reset_index(drop=True)
    os.makedirs(os.path.dirname(home_output_path), exist_ok=True)
    home_locations.to_csv(home_output_path, index=False)
    
    home_locations['home_flag'] = 1
    total_hours_df = total_hours_df.merge(home_locations[[DEVICE_ID_COL, LAT_COL, LON_COL, 'home_flag']], on=LOCATION_COLS, how='left')
    total_hours_df['home_flag'].fillna(0, inplace=True)

    if not home_only:
        workplace_output_path = os.path.join(base_path, 'home_workplace_location', f'workplace_location_{year}.csv')
        non_home_df = total_hours_df[total_hours_df['home_flag'] == 0]
        if not non_home_df.empty:
            idx_work = non_home_df.groupby(DEVICE_ID_COL)[DURATION_COL].idxmax()
            work_locations = non_home_df.loc[idx_work].reset_index(drop=True)
            work_locations.to_csv(workplace_output_path, index=False)

    print(f"--- Yearly processing for {year} finished ---")
    return total_hours_df 

# --- Part 3: Weekday/Weekend Analysis ---
# (保持原逻辑不变)
def analyze_and_save_day_counts_auto(year: int, base_path: str):
    print(f"Analyzing day counts for {year}...")
    file_pattern = os.path.join(base_path, 'home_workplace', f'*_{year}*_home_workplace.csv')
    file_list = glob(file_pattern)
    if not file_list: return None

    chunk_counts = []
    CHUNK_SIZE = 100
    for i in tqdm(range(0, len(file_list), CHUNK_SIZE)):
        chunk_files = file_list[i : i + CHUNK_SIZE]
        dfs = []
        for f in chunk_files:
            try:
                tmp = pd.read_csv(f, usecols=[DEVICE_ID_COL, 'day'], dtype={DEVICE_ID_COL: 'str', 'day': 'str'})
                dfs.append(tmp)
            except: continue
        if dfs:
            df_chunk = pd.concat(dfs, ignore_index=True).drop_duplicates()
            chunk_counts.append(df_chunk)
    
    if not chunk_counts: return None

    full_days_df = pd.concat(chunk_counts, ignore_index=True).drop_duplicates()
    full_days_df['dt'] = pd.to_datetime(full_days_df['day'], format='%Y%m%d', errors='coerce')
    full_days_df['is_weekend'] = (full_days_df['dt'].dt.dayofweek >= 5).astype(int)
    
    day_counts = full_days_df.groupby([DEVICE_ID_COL, 'is_weekend']).size().unstack(fill_value=0).reset_index()
    day_counts.rename(columns={0: 'day_count_weekday', 1: 'day_count_weekend'}, inplace=True)
    
    for col in ['day_count_weekday', 'day_count_weekend']:
        if col not in day_counts.columns: day_counts[col] = 0
            
    day_counts['day_count_all'] = day_counts['day_count_weekday'] + day_counts['day_count_weekend']
    
    output_file = os.path.join(base_path, 'home_workplace_location', f'weekday_weekend_count_{year}.csv')
    day_counts.to_csv(output_file, index=False)
    print(f"Saved automated day counts to {output_file}")
    return day_counts

# --- Part 4: Final Classification & Analysis (New & Optimized) ---

def get_valid_devices_cross_year(day_counts_2020: pd.DataFrame, day_counts_2021: pd.DataFrame, reference_year: int, measure_year: int, min_days: int = 4):
    """
    Identifies devices present in both years with sufficient data.
    """
    print(f"Identifying valid devices across {reference_year} and {measure_year}...")
    merged = day_counts_2020.merge(day_counts_2021, on=DEVICE_ID_COL, how='outer', suffixes=(f'_{reference_year}', f'_{measure_year}'))
    
    # Fill NA with 0 for calculation
    merged[[f'day_count_all_{reference_year}', f'day_count_all_{measure_year}']] = merged[[f'day_count_all_{reference_year}', f'day_count_all_{measure_year}']].fillna(0)
    
    # Filter logic
    valid_mask = (merged[f'day_count_all_{reference_year}'] >= min_days) & (merged[f'day_count_all_{measure_year}'] >= min_days)
    valid_devices = set(merged.loc[valid_mask, DEVICE_ID_COL].unique())
    
    output_path = os.path.join(HOME_WORK_LOC_DIR, f"weekday_weekend_count_{reference_year}_{measure_year}.csv")
    merged['device_aid_keep'] = valid_mask.astype(int)
    merged.to_csv(output_path, index=False)
    
    print(f"Found {len(valid_devices)} valid devices. Saved summary to {output_path}")
    return valid_devices

def process_and_classify_year(year: int, valid_devices: set, workplace_file_override=None, showed_hours_threshold: int = 6):
    """
    Loads daily files, filters by valid devices immediately (saving memory),
    merges with home/work locations, and classifies into Home/Work/Amenity.
    """
    print(f"\n--- Processing Final Classification for {year} ---")
    
    # 1. Load Reference Locations
    home_file = os.path.join(HOME_WORK_LOC_DIR, f'home_location_{year}.csv')
    
    # Workplace logic: Use specific year if available, otherwise fallback/override (per user logic using 2020 for both)
    if workplace_file_override:
        work_file = workplace_file_override
    else:
        work_file = os.path.join(HOME_WORK_LOC_DIR, f'workplace_location_{year}.csv')
        if not os.path.exists(work_file):
             # Fallback to 2020 if 2021 doesn't exist (mimicking user's logic)
             work_file = os.path.join(HOME_WORK_LOC_DIR, 'workplace_location_2020.csv')

    print(f"Loading reference locations:\n Home: {home_file}\n Work: {work_file}")
    
    try:
        home_df = pd.read_csv(home_file)
        work_df = pd.read_csv(work_file)
    except FileNotFoundError as e:
        print(f"Critical Error: Reference file not found. {e}")
        return

    # Add flags
    home_df['latitude'] = home_df['latitude'].astype('float32')
    home_df['longitude'] = home_df['longitude'].astype('float32')
    home_df['home_flag'] = 1

    work_df['latitude'] = work_df['latitude'].astype('float32')
    work_df['longitude'] = work_df['longitude'].astype('float32')
    work_df['workplace_flag'] = 1

    # 2. Iteratively Load and Filter Daily Data (Memory Optimized)
    daily_files = glob(os.path.join(OUTPUT_DATA_PATH, 'home_workplace', f'*_{year}_*_home_workplace.csv'))
    # Sort carefully (ensure matching part of filename is used for sort if needed, here just basic sort)
    daily_files.sort()
    
    filtered_chunks = []
    print(f"Loading and filtering {len(daily_files)} daily files...")
    
    for f in tqdm(daily_files):
        try:
            # 只读需要的列
            df = pd.read_csv(f, dtype=DTYPE_DICT)
            
            # 1. Filter by Hours
            if showed_hours_threshold is not None:
                df = df[df['showed_hours_today'] >= showed_hours_threshold]
            
            # 2. Filter by Valid Devices (Crucial for Memory)
            df = df[df[DEVICE_ID_COL].isin(valid_devices)]
            
            if df.empty: continue
            
            # Drop unnecessary columns to save memory before concat
            cols_to_drop = ['home', 'workplace'] # We will re-merge these from the master list
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            
            filtered_chunks.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
            
    if not filtered_chunks:
        print(f"No valid data found for {year}.")
        return

    # 3. Concatenate
    full_df = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks
    gc.collect()
    
    # Calculate weighted hours
    full_df['weighted_hours_today'] = (full_df[DURATION_COL] / full_df['showed_hours_today']) * 24
    
    # 4. Merge with Reference Locations
    # Left join to match specific coordinates
    merge_cols = [DEVICE_ID_COL, LAT_COL, LON_COL]
    
    full_df = full_df.merge(home_df[merge_cols + ['home_flag']], on=merge_cols, how='left')
    full_df = full_df.merge(work_df[merge_cols + ['workplace_flag']], on=merge_cols, how='left')
    
    # Fill NAs
    full_df['home_flag'] = full_df['home_flag'].fillna(0)
    full_df['workplace_flag'] = full_df['workplace_flag'].fillna(0)
    
    # 5. Classify
    # Create a 'place' column
    conditions = [
        (full_df['home_flag'] == 1),
        (full_df['workplace_flag'] == 1)
    ]
    choices = ['home', 'workplace']
    full_df['place'] = np.select(conditions, choices, default='amenity')
    
    # 6. Select Columns and Save
    output_cols = [DEVICE_ID_COL, LAT_COL, LON_COL, 'day', 'weighted_hours_today', 'showed_hours_today', 'place']
    final_df = full_df[output_cols]
    
    save_path = os.path.join(HOME_WORK_LOC_DIR, f'total_{year}_selected.csv')
    print(f"Saving final result to {save_path}...")
    final_df.to_csv(save_path, index=False)
    
    # 释放内存
    del full_df, final_df, home_df, work_df
    gc.collect()

def main():
    # 配置
    year_configs = [
        {"year": 2020, "home_only": False, "file_limits": None, "processed": True},
        # {"year": 2021, "home_only": True, "file_limits": None},
        {"year": 2022, "home_only": True, "file_limits": None, "processed": False}
    ]

    # --- Part 1: Process Files (Daily) ---
    all_files_to_process = []
    for config in year_configs:
        year = config["year"]
        if config.get("processed", True):
            continue
        file_pattern = INPUT_DATA_PATH_TEMPLATE.format(year=year)
        files_for_year = sorted(glob(file_pattern))[:config.get("file_limits", None)]
        all_files_to_process.extend(files_for_year) # 可在此处切片 [:10] 测试

    run_parallel_processing(all_files_to_process, OUTPUT_DATA_PATH)

    # --- Part 2 & 3: Yearly Aggregation & Day Counts ---
    day_counts_all = {}
    for config in year_configs:
        year = config["year"]
        if config.get("processed", True):
            day_counts = pd.read_csv(os.path.join(HOME_WORK_LOC_DIR, f'weekday_weekend_count_{year}.csv'))
        else:
            # 聚合
            process_yearly_data_optimized(year, OUTPUT_DATA_PATH, config["home_only"])
            
            # 统计天数
            day_counts = analyze_and_save_day_counts_auto(year, OUTPUT_DATA_PATH)
            if day_counts is not None:
                day_counts_all[year] = day_counts
        
        gc.collect()

    # day_counts_all[2020] = pd.read_csv(os.path.join(HOME_WORK_LOC_DIR, f'weekday_weekend_count_2020.csv'))
    # day_counts_all[2022] = pd.read_csv(os.path.join(HOME_WORK_LOC_DIR, f'weekday_weekend_count_2022.csv'))
    # --- Part 4: Cross-Year Validation & Final Classification (Refactored) ---
    
    # if 2020 in day_counts_all and 2021 in day_counts_all:
    if 2020 in day_counts_all and 2022 in day_counts_all:

        # 1. 获取有效设备 ID (Both years >= 4 days)
        valid_devices_set = get_valid_devices_cross_year(day_counts_all[2020], day_counts_all[2022], 2020, 2022)
        
        # 2. 处理 2020 数据
        # 注意：用户原代码中 2020 和 2021 都使用了 2020 的 workplace 文件。
        # 如果你想让 2021 使用它自己的 workplace 文件（如果存在），可以移除 workplace_file_override 参数
        workplace_file_2020 = os.path.join(HOME_WORK_LOC_DIR, 'workplace_location_2020.csv')
        
        process_and_classify_year(2020, valid_devices_set, workplace_file_override=workplace_file_2020)
        
        # 3. 处理 2021 数据
        # process_and_classify_year(2021, valid_devices_set, workplace_file_override=workplace_file_2020)

        process_and_classify_year(2022, valid_devices_set, workplace_file_override=workplace_file_2020)

        
    else:
        print("Skipping Part 4: Day counts data missing for 2020 or 2021.")

    print("\nFull pipeline finished.")


if __name__ == "__main__":
    main()

