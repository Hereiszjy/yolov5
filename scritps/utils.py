from tqdm import trange
from multiprocessing import Pool, cpu_count
import requests
import pandas as pd
import string
from pathlib import Path
from loguru import logger
import cv2 as cv
import numpy as np
from skimage import io
# from exif import Image
from PIL import ExifTags, Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os, json


HEX_DIGITS = set(string.hexdigits)
def url_to_name(url):
    name = Path(url).name
    assert not set(name).difference(HEX_DIGITS), name
    return name


def calc_iou(obj, ref):
    l, t, r, b = obj[:4]
    w = r - l
    h = b - t
    if w <= 0 or h <= 0:
        return 0

    rl, rt, rr, rb = ref[:4]
    rw = rr - rl
    rh = rb - rt
    if rw <= 0 or rh <= 0:
        return 0

    il = max(rl, l)
    it = max(rt, t)
    ir = min(rr, r)
    ib = min(rb, b)
    iw = ir - il
    ih = ib - it
    if iw <= 0 or ih <= 0:
        return 0

    itrs = iw * ih
    uni = rw * rh + w * h - itrs
    return itrs / uni


def extract_img_urls(label_file):
    annos = pd.read_csv(label_file, index_col=False, low_memory=False)
    url_keys = []
    for key in ['ImgUrl', 'ImageUrl']:
        if key in annos.keys():
            url_keys.append(key)
    # assert len(url_keys) == 1, f'Label contains more than one properties for image url: {url_keys}'
    key = url_keys[0]
    urls = annos[key].unique().tolist()
    del annos
    return urls


def extract_boxes(label_file, ignore_image_quality=True):
    keys = ['ImgUrl', 'ProductId', 'xmin', 'ymin', 'xmax', 'ymax']
    if not ignore_image_quality:
        keys.append('ImageQuality')
    annos = pd.read_csv(label_file, index_col=False, low_memory=False, usecols=keys)
    boxes = defaultdict(list)
    unqualified_names = set()
    for _, row in annos.iterrows():
        if ignore_image_quality:
            url, sku_id, l, t, r, b = [row[k] for k in keys]
        else:
            url, sku_id, l, t, r, b, quality = [row[k] for k in keys]
            quality = json.loads(quality)
            if quality:
                unqualified_names.add(name)
        name = url_to_name(url)
        sku_id = int(sku_id)
        l, t, r, b = [float(v) for v in [l, t, r, b]]
        boxes[name].append([sku_id, l, t, r, b])
    return {k: v for k, v in boxes.items() if k not in unqualified_names}


def download_image(req, timeout=30):
    url, path = req
    try:
        if path.is_file():
            return True
        jpeg = requests.get(url, timeout=timeout).content
        img = cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR)
        assert img is not None
        path.write_bytes(jpeg)
        return True
    except: # BaseException as e:
        return False


def download_images(download_list, num_proc=20):

    with ThreadPoolExecutor(num_proc) as pool:
        done = set()
        while len(done) != len(download_list):
            download_reqs = [[url, path] for url, path in download_list if url not in done]
            batch_num = (len(download_list) + num_proc - 1) // num_proc
            pbar = trange(batch_num)
            num_failed = 0
            for i in pbar:
                download_req_per_batch = download_reqs[i * num_proc : (i + 1) * num_proc]
                download_rsp_per_batch = pool.map(download_image, download_req_per_batch)
                for (url, _), success in zip(download_req_per_batch, download_rsp_per_batch):
                    if success:
                        done.add(url)
                    else:
                        num_failed += 1
                pbar.set_description(f'Downloading {len(download_reqs)} ({len(download_list)} in total), {num_failed} failed')


for ORIENTATION_INDEX in ExifTags.TAGS.keys():
    if ExifTags.TAGS[ORIENTATION_INDEX] == 'Orientation':
        break


def verify_image(img_path):
    ## Verify image corruption (does not work with cv2.imread() or PIL.Image.open())
    try:
        img = io.imread(img_path)
        h, w, _ = img.shape
        assert h > 0 and w > 0
        del img
    except:
        os.unlink(img_path)
        return 'Image corrupted'

    ## Verify orientation
    try:
        img = Image.open(img_path)
        orientation = dict(img._getexif().items())[ORIENTATION_INDEX]
        del img
        if orientation in [1]:
            return ''
        return f'Invalid orientation {orientation}'
    except:
        return ''
    assert False, 'BUG'


def verify_images(img_files, num_proc=20):
    batch_num = (len(img_files) + num_proc - 1) // num_proc
    pbar = trange(batch_num)
    num_errors = 0
    num_total = 0
    for i in pbar:
        with Pool(num_proc) as pool:
            img_files_per_batch = img_files[i * num_proc : (i + 1) * num_proc]
            num_total += len(img_files_per_batch)
            info_per_batch = pool.map(verify_image, img_files_per_batch)
            for path, error in zip(img_files_per_batch, info_per_batch):
                if error:
                    logger.error(f'{path}: {error}')
                    num_errors += 1
            pbar.set_description(f'Verified {num_total}, {num_errors} are corrupted')


def download_labqa(task_id, with_black_boxes):
    API_URL = 'https://labcanary.chinacloudsites.cn/api/services/app/task/ExportTaskResult'
    rsp = requests.post(API_URL, data={'TaskIds': [task_id], 'WithBlackBox': with_black_boxes})
    assert rsp.status_code == 200, '[%d] Invalid status code (%d)' % (task_id, rsp.status_code)
    content = rsp.json()
    assert 'success' in content and content['success'] == True, '[%d] Invalid response (%s)' % (task_id, content)
    assert 'error' in content and content['error'] is None, '[%d] Invalid response (%s)' % (task_id, content)
    assert 'result' in content and 'exportTaskResults' in content['result'], '[%d] Invalid response (%s)' % (task_id, content)
    assert len(content['result']['exportTaskResults']) == 1, '[%d] Invalid response (%s)' % (task_id, content)
    assert 'taskId' in content['result']['exportTaskResults'][0], '[%d] Invalid response (%s)' % (task_id, content)
    assert 'blobUrl' in content['result']['exportTaskResults'][0], '[%d] Invalid response (%s)' % (task_id, content)
    rsp_task_id = content['result']['exportTaskResults'][0]['taskId']
    label_url = content['result']['exportTaskResults'][0]['blobUrl']
    assert rsp_task_id == task_id, '[%d] Invalid response (%s)' % (task_id, content)
    rsp = requests.get(label_url)
    assert rsp.status_code == 200, '[%d] Invalid status code (%d)' % (task_id, rsp.status_code)
    return rsp.text
