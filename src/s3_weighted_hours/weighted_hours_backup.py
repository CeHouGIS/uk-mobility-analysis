import polars as pl
import pandas as pd
import pyreadr
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as csv

# ================= 配置部分 =================
YEAR = "2022"
BASE_PATH = Path("/nas/houce/UK_mobility/data_20251226_new/home_workplace_location")
if YEAR == "2021" or YEAR == "2022":
    MOBILITY_FILE = BASE_PATH / f"total_2020_selected_for_{YEAR}.csv"
elif YEAR == "2020_for_2021":
    MOBILITY_FILE = BASE_PATH / "total_2020_selected_for_2021.csv"
elif YEAR == "2020_for_2022":
    MOBILITY_FILE = BASE_PATH / "total_2020_selected_for_2022.csv"
else:
    raise ValueError("Unsupported YEAR value")

POI_FILE = BASE_PATH / "England_poi_2020_2021_2022_03.RData"
# OUTPUT_FILE = BASE_PATH / "amenity_weighted_hours_and_frequency_2020_for_2021.csv"
OUTPUT_FILE = BASE_PATH / f"amenity_weighted_hours_and_frequency_{YEAR}.csv"


POI_COLUMNS = {
    't1': 't1_eating_drinking',
    't3': 't3_attractions_entertainment',
    't5': 't5_health',
    't6': 't6_public_infrastructure',
    't8': 't8_reatail',
    't9': 't9_transport'
}

def load_and_clean_poi(filepath, year):
    """读取并清洗 POI 数据"""
    poi_data = pyreadr.read_r(str(filepath))
    # 自动获取第一个 key，防止硬编码 key 错误
    df_poi = poi_data[f"England_poi_{year}_03"]
    
    df_poi = df_poi.iloc[:-1].copy()
    # 确保 location 是字符串类型，方便后续 merge
    if 'location' in df_poi.columns:
        df_poi['location'] = df_poi['location'].astype(str)
    return df_poi

def process_mobility_data(mobility_path, poi_df):
    """处理流动性数据并计算加权指标"""
    
    # 1. 使用 PyArrow 高效读取
    # 修正：csv.read_csv 没有 ncols 参数，应该使用 read_options
    # 解决 Offset Overflow: 使用 large_string 并增加块大小
    # convert_options = csv.ConvertOptions(
    #     column_types={col: pa.large_string() for col in ['device_aid', 'place', 'latitude', 'longitude']}
    # )
    
    # print(f"Reading {mobility_path}...")
    # table = csv.read_csv(mobility_path, convert_options=convert_options)

    # # 转换回 pandas (推荐暂不使用 ArrowDtype 除非环境支持极好)
    # # 为了计算性能，我们将数值列转回 numpy
    # df = table.to_pandas()

    pl.read_csv(mobility_path).write_parquet("data.parquet")

    # 以后读取只需要 几秒钟
    df = pl.read_parquet("data.parquet")

    # 2. 筛选数据
    # 先做初步筛选，减少内存占用
    # Polars 筛选
    df = df.filter(pl.col('place') == 'amenity')
    # 转换回 pandas 以兼容后续代码
    df = df.to_pandas()
    

    max_amenity_idx = df[df['place'] == 'amenity'].groupby(['device_aid', 'day'])['weighted_hours_today'].idxmax()
    print("Saving max amenity travel data...")
    df.loc[max_amenity_idx].to_csv(f"/nas/houce/UK_mobility/data_20251226_new/home_workplace_location/travel_uk_{YEAR}.csv", index=False)

    # 确保数值列类型正确
    num_cols = ['weighted_hours_today', 'showed_hours_today', 'latitude', 'longitude']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 过滤并增加频率计数
    df_filtered = df[df['showed_hours_today'] > 11].copy()
    df_filtered['frequency'] = 1.0


    # 4. 构建连接键 (注意精度控制)
    # 直接使用 string 会导致 0.1 和 0.10000 匹配不上，建议在读取时统一格式
    df_filtered['location'] = (
        df_filtered['latitude'].round(5).astype(str) + "_" + 
        df_filtered['longitude'].round(5).astype(str)
    )
    # 同样对 POI 做精度处理 (如果 POI 也是经纬度拼接的话)
    

    # 5. 合并 POI 数据
    # 优化：先过滤再 merge
    merged_df = df_filtered.merge(poi_df, on='location', how='inner')
    
    # 排除分母为 0 的情况
    merged_df = merged_df[merged_df['poi_total'] > 0].copy()

    # 6. 批量计算加权指标
    agg_dict = {}
    for prefix, col_name in POI_COLUMNS.items():
        # 确保 POI 权重列是数值
        merged_df[col_name] = pd.to_numeric(merged_df[col_name], errors='coerce').fillna(0)
        
        wh_col = f'{prefix}_weighted_hours_today'
        freq_col = f'{prefix}_frequency'
        
        # 计算
        weight_factor = merged_df[col_name] / merged_df['poi_total']
        merged_df[wh_col] = merged_df['weighted_hours_today'] * weight_factor
        merged_df[freq_col] = merged_df['frequency'] * weight_factor
        
        agg_dict[wh_col] = 'sum'
        agg_dict[freq_col] = 'sum'

    # 7. 分组聚合
    result = merged_df.groupby(['device_aid', 'day']).agg(agg_dict).reset_index()
    
    return result

def main():
    print("Step 1: Loading POI Data...")
    df_poi = load_and_clean_poi(POI_FILE, YEAR)
    
    print("Step 2: Processing Mobility Data...")
    final_df = process_mobility_data(MOBILITY_FILE, df_poi)
    
    print(f"Step 3: Saving results to {OUTPUT_FILE}...")
    # 检查输出目录是否存在
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Finished successfully.")

if __name__ == "__main__":
    main()