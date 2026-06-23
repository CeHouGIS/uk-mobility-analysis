# python src/s3_weighted_hours/weighted_hours.py --year 2022
import polars as pl
import pandas as pd
import pyreadr
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as csv
import argparse
import sys
import os

# ================= 配置常量 =================
POI_COLUMNS = {
    't1': 't1_eating_drinking',
    't3': 't3_attractions_entertainment',
    't5': 't5_health',
    't6': 't6_public_infrastructure',
    't8': 't8_reatail',
    't9': 't9_transport'
}

def get_args():
    parser = argparse.ArgumentParser(description="Process UK mobility and POI data.")
    
    # 核心参数：年份
    parser.add_argument(
        "--year", 
        type=str, 
        default="2022",
        choices=["2021", "2022", "2020_for_2021", "2020_for_2022"],
        help="The year to process (default: 2022)"
    )
    
    # 路径参数
    parser.add_argument(
        "--base_path", 
        type=Path, 
        default=Path("/nas/houce/UK_mobility/data_20251226_new/home_workplace_location"),
        help="Base directory for input files"
    )
    
    parser.add_argument(
        "--output", 
        type=Path, 
        help="Output CSV path (if not set, generated based on year)"
    )

    return parser.parse_args()

def resolve_files(args):
    """根据输入的 year 和 base_path 确定具体文件路径"""
    year = args.year
    base_path = args.base_path
    
    # 匹配流动性文件逻辑
    if year in ["2021", "2022"]:
        mobility_file = base_path / f"total_{year}_selected.csv"
    elif year == "2020_for_2021":
        mobility_file = base_path / "total_2020_selected_for_2021.csv"
    elif year == "2020_for_2022":
        mobility_file = base_path / "total_2020_selected_for_2022.csv"
    else:
        # 虽然 argparse 有 choices，但这里加一层保护
        raise ValueError(f"Unsupported YEAR value: {year}")

    poi_file = base_path / "England_poi_2020_2021_2022_03.RData"
    
    # 确定输出文件
    if args.output:
        output_file = args.output
    else:
        os.makedirs(base_path / "amenity_weighted_hours_and_frequency", exist_ok=True)
        output_file = base_path / "amenity_weighted_hours_and_frequency" / f"amenity_weighted_hours_and_frequency_{year}.csv"
        
    # 中间产生的 travel_uk 文件路径
    travel_save_path = base_path / "amenity_weighted_hours_and_frequency" / f"travel_uk_{year}.csv"

    return mobility_file, poi_file, output_file, travel_save_path

def load_and_clean_poi(filepath, year):
    """读取并清洗 POI 数据"""
    # 注意：RData 内部的 key 可能只取年份数字部分，如 2020_for_2021 对应 2021
    clean_year = year.split('_')[-1] 
    
    poi_data = pyreadr.read_r(str(filepath))
    key = f"England_poi_{clean_year}_03"
    
    if key not in poi_data:
        # 如果找不到精确匹配，打印所有 key 辅助调试
        print(f"Warning: Key {key} not found. Available keys: {list(poi_data.keys())}")
        key = list(poi_data.keys())[0]

    df_poi = poi_data[key]
    df_poi = df_poi.iloc[:-1].copy()
    
    if 'location' in df_poi.columns:
        df_poi['location'] = df_poi['location'].astype(str)
    return df_poi

def process_mobility_data(mobility_path, poi_df, travel_save_path):
    """处理流动性数据并计算加权指标"""
    
    print(f"Converting {mobility_path} to Parquet for speed...")
    # 使用 Polars 加速读取和预过滤
    # 增加 infer_schema_length 防止类型推断错误
    pl.read_csv(mobility_path, infer_schema_length=10000).write_parquet("temp_data.parquet")
    
    df_pl = pl.read_parquet("temp_data.parquet")
    df_pl = df_pl.filter(pl.col('place') == 'amenity')
    
    # 转换回 pandas 处理后续复杂逻辑
    df = df_pl.to_pandas()

    # 计算并保存 max amenity 记录
    print("Saving max amenity travel data...")
    max_amenity_idx = df.groupby(['device_aid', 'day'])['weighted_hours_today'].idxmax()
    df.loc[max_amenity_idx].to_csv(travel_save_path, index=False)

    # 数值转换
    num_cols = ['weighted_hours_today', 'showed_hours_today', 'latitude', 'longitude']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 过滤
    df_filtered = df[df['showed_hours_today'] > 11].copy()
    df_filtered['frequency'] = 1.0

    # 构建连接键
    df_filtered['location'] = (
        df_filtered['latitude'].astype(str) + "_" + 
        df_filtered['longitude'].astype(str)
    )

    # 合并 POI
    merged_df = df_filtered.merge(poi_df, on='location', how='inner')
    merged_df = merged_df[merged_df['poi_total'] > 0].copy()

    # 批量计算
    agg_dict = {}
    for prefix, col_name in POI_COLUMNS.items():
        if col_name not in merged_df.columns:
            continue
            
        merged_df[col_name] = pd.to_numeric(merged_df[col_name], errors='coerce').fillna(0)
        
        wh_col = f'{prefix}_weighted_hours_today'
        freq_col = f'{prefix}_frequency'
        
        weight_factor = merged_df[col_name] / merged_df['poi_total']
        merged_df[wh_col] = merged_df['weighted_hours_today'] * weight_factor
        merged_df[freq_col] = merged_df['frequency'] * weight_factor
        
        agg_dict[wh_col] = 'sum'
        agg_dict[freq_col] = 'sum'

    # 聚合
    result = merged_df.groupby(['device_aid', 'day']).agg(agg_dict).reset_index()
    return result

def main():
    args = get_args()
    
    try:
        mobility_file, poi_file, output_file, travel_save_path = resolve_files(args)
        
        print(f"--- Configuration ---")
        print(f"Year: {args.year}")
        print(f"Input: {mobility_file}")
        print(f"POI: {poi_file}")
        print(f"Output: {output_file}")
        print(f"----------------------")

        print("Step 1: Loading POI Data...")
        df_poi = load_and_clean_poi(poi_file, args.year)
        
        print("Step 2: Processing Mobility Data...")
        final_df = process_mobility_data(mobility_file, df_poi, travel_save_path)
        
        print(f"Step 3: Saving results to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_file, index=False)
        
        print("Finished successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()