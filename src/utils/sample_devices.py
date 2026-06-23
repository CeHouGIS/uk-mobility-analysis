# python src/utils/sample_devices.py
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import pyreadr
import random

# aaa = pd.read_csv("/nas/houce/UK_mobility/processed_data/uk_2021_0331_processed.csv")

ref_ids_csv = pyreadr.read_r("/nas/houce/UK_mobility/individual mobility profile previous/device_aid_keep_England.Rdata")
ref_ids = ref_ids_csv['device_aid_keep']['device_aid_keep'].tolist()
select_num = round(0.1 * len(ref_ids))
sampled_ref_ids = random.sample(ref_ids, select_num)

# ref_ids_csv = pd.read_csv("/nas/houce/UK_mobility/individual mobility profile previous/England_individual_1127.csv")
# ref_ids = ref_ids_csv['device_aid'].values
save_path = "/nas/houce/UK_mobility/processed_data_selected/"

file_paths = glob("/nas/houce/UK_mobility/processed_data/*_processed.csv")
print(f"Found {len(file_paths)} files to process.")
for mob_file in tqdm(file_paths):
    print(f"Processing file: {mob_file}")
    file_basename = os.path.basename(mob_file)
    df = pd.read_csv(mob_file)
    df_selected = df[df['device_aid'].isin(sampled_ref_ids)]
    df_selected = df_selected[df_selected['showed_hours_today'] > 7].reset_index()
    df_selected.to_csv(os.path.join(save_path, f"{file_basename[:-4]}_selected_{len(sampled_ref_ids)}.csv"), index=False)
    print(f"Selected {len(df_selected)} rows. Saving to: {os.path.join(save_path, f'{file_basename[:-4]}_selected_{len(sampled_ref_ids)}.csv')}")
    print(f"Finished processing {file_basename}.\n")