# python /code/Mob/filter_ori_files.py
from glob import glob
import pandas as pd
import os
import pyreadr
from tqdm import tqdm

data_paths = glob("/nas/houce/UK_mobility/*/UK_2020_0201.*")

for i, data_path in tqdm(enumerate(data_paths), total=len(data_paths)):
    if data_path.endswith('.csv'):
        continue
    print(f"当前处理文件: {data_path}")
    # if '2020 Feb' in data_path or '2020 March' in data_path or '2021 March' in data_path:
    #     continue
    year = data_path.split('/')[-1].split('_')[1]
    date = data_path.split('/')[-1].split('_')[2].split('.')[0]
    print(f"年份: {year}, 日期: {date}")
    
    result = pyreadr.read_r(data_path)

    data = list(result.values())[0]  # 获取第一个数据框

    if ('2020 Feb' in data_path) or ('2022 March' in data_path):
        print("处理2020 Feb数据，添加day和hours字段...")
        data['day'] = data['hour'].apply(lambda x: int(int(f"{int(x.split(' ')[0].split('-')[0]):04d}{int(x.split(' ')[0].split('-')[1]):02d}{int(x.split(' ')[0].split('-')[2]):02d}")))
        data['hours'] = data['hour'].apply(lambda x: int(x.split(' ')[1]))

    device_counts = data.groupby(['device_aid', 'day', 'hours']).size().reset_index(name='count')
    device_day_summary = device_counts.groupby('device_aid').agg(
        total_records=(f'count', 'sum'),
        hours_present=(f'hours', 'nunique')
    ).reset_index()
    device_day_summary = device_day_summary.rename(columns={
        'total_records': f'total_records_{year}{date}',
        'hours_present': f'hours_present_{year}{date}'
    })

    filter_day = device_day_summary[device_day_summary[f'hours_present_{year}{date}'] > 11]

    print(f"保存筛选信息到：/nas/houce/UK_mobility/processed_data/filter_day_info_{year}{date}.csv")
    filter_day.to_csv(rf'/nas/houce/UK_mobility/processed_data/filter_day_info_{year}{date}.csv', index=False)
    filter_id_list = filter_day[filter_day[f'hours_present_{year}{date}'] >= 11]['device_aid'].tolist()
    print(f"保存筛选后的原始数据到：/nas/houce/UK_mobility/processed_data/filtered_data_{year}{date}.csv")
    data[data['device_aid'].isin(filter_id_list)].to_csv(fr'/nas/houce/UK_mobility/processed_data/filtered_data_{year}{date}.csv', index=False)