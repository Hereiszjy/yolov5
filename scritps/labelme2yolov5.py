from argparse import ArgumentParser
from tqdm import tqdm, trange
from collections import defaultdict
import sys, shutil
import numpy as np
from pathlib import Path
import cv2 as cv
from loguru import logger
import functools
from multiprocessing import Pool
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils import extract_boxes, calc_iou


ROTATE_ID = 4347684
IGNORE_AREA_ID = 1047936
# SKU_IDS = [ROTATE_ID, IGNORE_AREA_ID]
SKU_IDS = [4361889,4361890,4361891,4361892,4361902,4361905,4361906,4361907,4361909,4373780,4373781,4373783,4374091,4374197]
NAMES = {ROTATE_ID: 'ROTATE', IGNORE_AREA_ID: 'IGNORE_AREA'}
COLORS = {ROTATE_ID: (0, 255, 0), IGNORE_AREA_ID: (0, 0, 255)}
VISUALIZE = False
DEDUPE = True


def debug(img_file, boxes):
    img = load_img(str(img_file))
    H, W = img.shape[:2]
    for sku_id, l, t, r, b in boxes:
        cv.rectangle(img, (int(l * W), int(t * H)), (int(r * W), int(b * H)), COLORS[sku_id], 3)
    cv.imshow(f'{sku_id}', img)
    if cv.waitKey() == 27:
        exit()
    cv.destroyAllWindows()


def convert_coords(l, t, r, b):
    assert l >= 0 and l < r and r <= 0.5
    assert t >= 0 and t < b and b <= 1
    assert t >= 0.5 or b <= 0.5
    top = b <= 0.5
    if not top:
        t, b = t - 0.5, b - 0.5
    return top, (2 * l, 2 * t, 2 * r, 2 * b)


def has_overlap(box, boxes):
    for ref in boxes:
        if calc_iou(ref, box) > 0.5:
            return True
    return False


def load_img(img_file, max_size=640):
    img = cv.imread(str(img_file))
    assert img is not None, img_file
    h, w = img.shape[:2]
    assert h % 2 == 0 and w % 2 == 0, (w, h)
    h, w = h // 2, w // 2
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / max(h, w)
        h, w = [round(scale * v) for v in [h, w]]
        return cv.resize(img, (2 * w, 2 * h))
    return img


def process(name, boxes_by_name, raw_img_dir, img_dir, lbl_dir, vis_dir):

    lbl_file = lbl_dir / f'{name}.txt'
    img_file = img_dir / f'{name}.jpg'
    boxes_pre_sku = boxes_by_name[name]

    cnt_by_sku = defaultdict(int)

    boxes, annos = [], []
    
    for sku_id, l, t, r, b in boxes_pre_sku:

        model_id = SKU_IDS.index(sku_id)
        x, y = 0.5 * (l + r), 0.5 * (t + b)
        w, h = r - l, b - t
        boxes.append([l, t, r, b, model_id])
        annos.append(f'{model_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
        cnt_by_sku[sku_id] += 1

    lbl_file.write_text('\n'.join(annos))

    return (str(img_file)), cnt_by_sku



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('--num_proc', type=int, default=1)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    raw_img_dir = root_dir / 'images'
    raw_lbl_dir = root_dir / 'raw_labels'
    dataset_dir = root_dir / 'dataset'
    img_dir = dataset_dir / 'images'
    lbl_dir = dataset_dir / 'labels'
    vis_dir = dataset_dir / 'vis'

    shutil.rmtree(dataset_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        f'train: {dataset_dir}/train.list',
        f'val: {dataset_dir}/val.list',
        'nc: 14', 'names: [4361889,4361890,4361891,4361892,4361902,4361905,4361906,4361907,4361909,4373780,4373781,4373783,4374091,4374197]'
    ]
    config_file = dataset_dir / 'dataset.yaml'
    config_file.write_text('\n'.join(configs))

    ## Enumerate raw label files
    raw_lbl_files = list(raw_lbl_dir.glob('*.csv'))
    
    ## Load boxes
    boxes = {}
    names = set()
    num_dup_imgs, num_dup_boxes = 0, 0
    for raw_lbl_file in raw_lbl_files:
        boxes_per_file = extract_boxes(raw_lbl_file, ignore_image_quality=False)
        for name in boxes_per_file:
            if name in names:
                num_dup_imgs += 1
                num_dup_boxes += len(boxes_per_file[name])
                continue
            names.add(name)
            boxes[name] = boxes_per_file[name]

    ## Generate dataset
    all_img_files = []
    cnt_by_sku = defaultdict(int)
    if args.num_proc > 1:
        pool = Pool(args.num_proc)
        names_per_phase = list(boxes.keys())
        num_batch = (len(boxes) + args.num_proc - 1) // args.num_proc
        for i in trange(num_batch):
            names_per_batch = names_per_phase[args.num_proc * i : args.num_proc * (i + 1)]
            assert names_per_batch
            result = pool.map(functools.partial(process, boxes_by_name=boxes, raw_img_dir=raw_img_dir, img_dir=img_dir, lbl_dir=lbl_dir, vis_dir=vis_dir), names_per_batch)
            for files, cnt in result:
                all_img_files.append(files)
                for sku, num in cnt.items():
                    cnt_by_sku[sku] += num
        pool.close()
    else:
        for name in tqdm(boxes):
            img_files, cnt = process(name, boxes, raw_img_dir, img_dir, lbl_dir, vis_dir)
            all_img_files.append(img_files)
            for sku, num in cnt.items():
                cnt_by_sku[sku] += num

    list_file = dataset_dir / 'all.list'
    list_file.write_text('\n'.join(all_img_files))
    
    #随机打乱以便区分训练集和验证集
    np.random.shuffle(all_img_files)
    val_num = round(0.1 * len(all_img_files))
    train_num = len(all_img_files) - val_num
    
    train_list_file = dataset_dir / 'train.list'
    train_list_file.write_text('\n'.join(all_img_files[:train_num]))
    val_list_file = dataset_dir / 'val.list'
    val_list_file.write_text('\n'.join(all_img_files[train_num:]))

    logger.info(f'==> Generated {len(all_img_files)} images')

