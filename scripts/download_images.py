import sys
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger
sys.path.append(str(Path(__file__).absolute().parent))
from utils import extract_img_urls, url_to_name, download_images, verify_images


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--work_dir')
    args = parser.parse_args()

    ## Generate download list
    work_dir = Path(args.work_dir)
    raw_label_dir = work_dir / 'raw_labels'
    label_paths = list(raw_label_dir.rglob('*.csv'))
    logger.info(f'Found {len(label_paths)} raw labels under {raw_label_dir}')
    img_urls = []
    for label_path in tqdm(label_paths):
        img_urls.extend(extract_img_urls(label_path))
    url_by_name = {url_to_name(url): url for url in img_urls}
    del img_urls
    logger.info(f'Found {len(url_by_name)} images to download')

    ## Download images
    img_dir = work_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Downloading {len(url_by_name)} images to {img_dir} ...')
    download_list = [[url, img_dir / f'{name}.jpg'] for name, url in url_by_name.items()]
    download_images(download_list)
    del download_list
