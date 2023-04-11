# 本脚本用于合并csv，并清理

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

raw_label_dir = Path('/data/disk3/cv2022/data/shandong_gravity/raw_labels')
label_paths = list(raw_label_dir.rglob('*.csv'))



merge = pd.DataFrame(columns=['ImgUrl','ImageQuality','ProductId','xmin','ymin','xmax','ymax'])

for label_path in tqdm(label_paths):
    df = pd.read_csv(str(label_path),usecols=['ImgUrl','ImageQuality','ProductId','xmin','ymin','xmax','ymax'])

    df_sub = df.loc[df["ProductId"]==4347684]

    merge = pd.concat([merge,df_sub])

merge.to_csv(raw_label_dir/'merge.csv', index=False)