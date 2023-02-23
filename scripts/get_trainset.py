# 本脚本用于从全量数据集中筛选出目标sku的crop，用于特定的训练任务

import pandas as pd
from pathlib import Path
from tqdm import tqdm
subdataset = 'ccth_exp003_img1000'
dataset_dir = '/home/jiayu/yolov5/dataset/ccth_train_data.csv'
skulist_dir = f'/home/jiayu/yolov5/sku_list/{subdataset}.csv'
raw_labels_dir = Path(f'/data/disk3/cv2022/data/{subdataset}/raw_labels')
raw_labels_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(dataset_dir, usecols=[0,2,3,4,5,6])
skuID = pd.read_csv(skulist_dir)['ProductId'].values.tolist()
skuID.sort()

sub = pd.DataFrame(columns=['ImgUrl','ProductId','xmin','ymin','xmax','ymax'])
# sample 500 imgs pre sku
for sku in tqdm(skuID):
    sku_rows = df.loc[df["ProductId"]==sku]
    img_list = sku_rows["ImgUrl"].values.tolist()[:1000]
    sub_sku = sku_rows[sku_rows.ImgUrl.isin(img_list)]
    sub = pd.concat([sub,sub_sku])

sub.to_csv(raw_labels_dir/'sub.csv', index=False)