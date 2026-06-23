# python /code/Mob/ML.py

from glob import glob
import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm

for file_path in glob("/nas/houce/UK_mobility/output_data/*"):
    print(f"Processing files: {file_path}")
    data_counts = pd.read_csv(file_path)

    time1 = file_path.split('/')[-1].split('_')[-2]
    time2 = file_path.split('/')[-1].split('_')[-1].split('.')[0]
    print(time1, time2)

    cols_1 = [col for col in data_counts.columns if col.startswith(f'hours_present_{time1}')]
    cols_2 = [col for col in data_counts.columns if col.startswith(f'hours_present_{time2}')]
    filtered = data_counts[(data_counts[cols_1].gt(11).any(axis=1)) & (data_counts[cols_2].gt(11).any(axis=1))]

    # 筛选在202103和202203两个时间段内出现天数都大于5天的数据
    mask_1 = data_counts[cols_1].gt(5).any(axis=1)
    mask_2 = data_counts[cols_2].gt(5).any(axis=1)
    result = data_counts[mask_1 & mask_2]

    for row_timepoint in tqdm(result.columns[1:]):
        print(f"\nprocessing time point: {row_timepoint}")
        timepoint = row_timepoint.split('_')[-1]
        print(f"current time point: {timepoint}")
        ids_list = result[result[row_timepoint].notna()]['device_aid'].tolist()
        print(f"Filter {len(ids_list)} valid IDs")
        amenity_detection_filepath = f"/nas/houce/UK_mobility/amenity_detection/amenity_detection_{timepoint}"
        all_amenity_detection_files = glob(amenity_detection_filepath + '/*.csv')
        output_path = f"/nas/houce/UK_mobility/selected_amenity_detection_ids/{time1}_{time2}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Create output path: {output_path}")
        for file in tqdm(all_amenity_detection_files):
            df = pd.read_csv(file)
            if 'hour' in df.columns:
                df = df.drop(columns='hour')
            df_filtered = df[df['device_aid'].isin(ids_list)]
            for i, (name, group_df) in tqdm(enumerate(df_filtered.groupby('device_aid')), total=df_filtered['device_aid'].nunique()):
                if os.path.exists(os.path.join(output_path, f"selected_ids_{name}.csv")):
                    print(f"file exists, continue write in: selected_ids_{name}.csv")
                    group_df.to_csv(os.path.join(output_path, f"selected_ids_{name}.csv"), mode='a', header=False, index=False)
                else:   
                    group_df.to_csv(os.path.join(output_path, f"selected_ids_{name}.csv"), index=False)