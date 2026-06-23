# python src/s2_amenity/detect_amenity_stops.py

from glob import glob
import pandas as pd
import os
import pyreadr
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings
import time
warnings.filterwarnings("ignore")

def process_user_data(user_tuple, time_):
    """处理单个用户的数据"""
    try:
        i, (device_id, df) = user_tuple
        
        # 计算停留时间
        df['duration'] = 60 / df.groupby('hours')['hours'].transform('count')
        df['duration'] = df['duration'].round(2)
        
        # 复制数据而不是使用merge_duplicates函数
        df_merged = df.copy()
        # print(f"User {i}, ID: {device_id}, Records: {len(df)}")
        df_merged['status'] = 'moving'  # 默认都是移动状态
        
        # 获取前一行的坐标
        df_merged['prev_latitude'] = df_merged['latitude'].shift()
        df_merged['prev_longitude'] = df_merged['longitude'].shift()
        
        # 如果duration > 5，标记为兴趣点
        df_merged.loc[(df_merged['duration'] > 5), 'status'] = 'poi'
        
        # 如果坐标与前一小时相同，标记为兴趣点
        df_merged.loc[(df_merged['latitude'] == df_merged['prev_latitude']) & 
            (df_merged['longitude'] == df_merged['prev_longitude']), 'status'] = 'poi'
        
        # 删除临时计算列
        df_merged = df_merged.drop(columns=['prev_latitude', 'prev_longitude'])
        
        # 只保留兴趣点
        df_merged = df_merged[df_merged['status'] != 'moving']
        
        # 标记兴趣点
        location_poi_label = pd.DataFrame(df_merged[df_merged['status'] == 'poi'].value_counts(['latitude','longitude'])).reset_index()
        location_poi_label.columns = ['latitude', 'longitude', 'count']
        location_poi_label['label'] = 'amenity'
        
        # 设置家和工作地点
        if len(location_poi_label) > 0:
            location_poi_label.loc[0, 'label'] = 'home'
        if len(location_poi_label) > 1:
            location_poi_label.loc[1, 'label'] = 'workplace'
            
        df_merged = df_merged.merge(location_poi_label, on=['latitude', 'longitude'], how='left')
        # df_merged = df_merged.drop_duplicates(subset=['latitude', 'longitude', 'label'])
        
        return df_merged
    except Exception as e:
        print(f"Error processing user {device_id}: {str(e)}")
        return pd.DataFrame()

def main():
    # 获取CPU核心数
    num_cpus = multiprocessing.cpu_count()
    print(f"检测到 {num_cpus} 个CPU核心，将使用 {max(1, num_cpus-1)} 个核心进行处理")
    
    # 文件列表
    file_paths = glob("/nas/houce/UK_mobility/output_data/merged_*.csv")
    print(f"找到 {len(file_paths)} 个合并文件需要处理")
    
    for file_idx, file_path in enumerate(file_paths):
        print(f"处理文件 {file_idx+1}/{len(file_paths)}: {file_path}")
        time1 = file_path.split('/')[-1].split('_')[1]
        time2 = file_path.split('/')[-1].split('_')[2].split('.')[0]
        print(f"解析时间段: {time1} 和 {time2}")
        
        # 读取数据
        print("读取合并数据...")
        data_counts = pd.read_csv(file_path)
        print(f"合并数据形状: {data_counts.shape}")

        # 筛选列和数据
        cols_1 = [col for col in data_counts.columns if col.startswith(f'hours_present_{time1}')]
        cols_2 = [col for col in data_counts.columns if col.startswith(f'hours_present_{time2}')]
        print(f"筛选条件: {time1}期间出现>11小时 且 {time2}期间出现>11小时")
        
        filtered = data_counts[(data_counts[cols_1].gt(11).any(axis=1)) & (data_counts[cols_2].gt(11).any(axis=1))]
        print(f"筛选后设备数量: {len(filtered)}")
        
        time_all = [i.split('_')[-1] for i in filtered.columns[1:] if i.startswith('hours_present_')]
        print(f"找到 {len(time_all)} 个时间段")

        for j, time_ in enumerate(time_all):
            print(f"\n处理时间段 {j+1}/{len(time_all)}: {time_}")
            data_file = f"/nas/houce/UK_mobility/processed_data/filtered_data_{time_}.csv"
            
            if not os.path.exists(data_file):
                print(f"文件不存在: {data_file}")
                continue
                
            print(f"读取文件: {data_file}")
            data = pd.read_csv(data_file)
            print(f"原始数据形状: {data.shape}")

            # 筛选ID
            id_list = filtered[filtered[f'hours_present_{time_}'].notna()]['device_aid'].tolist()
            print(f"符合条件的设备ID数量: {len(id_list)}")
            
            data_filtered = data[data['device_aid'].isin(id_list)]
            print(f"筛选后数据形状: {data_filtered.shape}")
            
            # 按设备ID分组
            data_filtered_group = data_filtered.groupby('device_aid')
            print(f"分组后用户数量: {len(data_filtered_group)}")
            
            # 创建输出目录
            output_dir = f"/nas/houce/UK_mobility/amenity_detection/amenity_detection_{time_}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 分批处理用户
            batch_size = 2000
            total_users = len(data_filtered_group)
            batches = [range(i, min(i+batch_size, total_users)) for i in range(0, total_users, batch_size)]
            
            print(f"将 {total_users} 个用户分为 {len(batches)} 批处理，每批 {batch_size} 个用户")
            
            for batch_idx, batch_range in enumerate(batches):
                start_i = batch_range[0]
                end_i = batch_range[-1]
                print(f"\n处理批次 {batch_idx+1}/{len(batches)}: 用户 {start_i} 到 {end_i}")
                
                # 获取当前批次的用户数据
                batch_users = [item for i, item in enumerate(data_filtered_group) if i in batch_range]
                
                # 使用多进程处理
                results = []
                with ProcessPoolExecutor(max_workers=max(1, num_cpus-1)) as executor:
                    futures = {executor.submit(process_user_data, (idx, user_data), time_): idx 
                              for idx, user_data in enumerate(batch_users, start=start_i)}
                    
                    # 收集结果
                    for future in tqdm(futures, desc=f"批次 {batch_idx+1} 处理进度"):
                        try:
                            result = future.result()
                            if not result.empty:
                                results.append(result)
                        except Exception as e:
                            print(f"处理出错: {str(e)}")
                
                # 合并并保存结果
                if results:
                    df_merged_all = pd.concat(results, ignore_index=True)
                    output_file = f"{output_dir}/batch_{start_i}_{end_i}.csv"
                    print(f"保存批次结果到: {output_file}, 记录数: {len(df_merged_all)}")
                    df_merged_all.to_csv(output_file, index=False)
                else:
                    print(f"批次 {batch_idx+1} 没有有效结果")
                
                # 释放内存
                del results
                
            print(f"时间段 {time_} 处理完成\n")
            
        print(f"文件 {file_path} 处理完成\n")
    
    print("所有文件处理完成")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"总运行时间: {elapsed_time/60:.2f} 分钟")