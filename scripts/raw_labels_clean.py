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

    ## Generate download list
    work_dir = Path(args.work_dir)
    raw_label_dir = work_dir / 'raw_labels_dirty'
    label_paths = list(raw_label_dir.rglob('*.csv'))
    logger.info(f'Found {len(label_paths)} raw labels under {raw_label_dir}')
    
    # clean and merge csv
    result = pd.DataFrame(columns=['ImgUrl','ImageQuality','ProductId','xmin','ymin','xmax','ymax'])
    for label_path in tqdm(label_paths):
        current = pd.read_csv(label_path, usecols=['ImgUrl','ImageQuality','ProductId','xmin','ymin','xmax','ymax'])
        current = current.loc[current['ProductId']==4347684]
        current = current.loc[current["ImageQuality"]=='[]']
        result = pd.concat([result, current])
    
    result.to_csv('merged.csv', index=False)