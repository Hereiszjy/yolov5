# 删除数据集中部分POSM，并合并成新的csv

import sys
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger
sys.path.append(str(Path(__file__).absolute().parent))
import pandas as pd

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--work_dir')
    args = parser.parse_args()
    DELETE_SKU = [4361890, 4361892, 4361900, 4361907, 4373783, 4374091]
    ## Generate download list
    work_dir = Path(args.work_dir)
    raw_label_dir = work_dir / 'raw_labels'
    label_paths = list(raw_label_dir.rglob('*.csv'))
    logger.info(f'Found {len(label_paths)} raw labels under {raw_label_dir}')
    
    result = pd.DataFrame(columns=['ImgUrl','ProductId','xmin','ymin','xmax','ymax'])
    for label_path in tqdm(label_paths):
        label = pd.read_csv(label_path, usecols=['ImgUrl','ProductId','xmin','ymin','xmax','ymax'])
        sub = label.loc[~label["ProductId"].isin(DELETE_SKU)]
        result = pd.concat([result, sub])
    result.to_csv(str(work_dir / 'raw_labels'/'merged0316.csv'), index=False)







