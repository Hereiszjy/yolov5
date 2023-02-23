import os
from glob import glob
import argparse
import cv2 as cv
import numpy as np
import shutil
from tqdm import tqdm
from terminaltables import AsciiTable
from collections import defaultdict
import json
import pandas as pd


def calc_iou(obj, area):
    l1, t1, r1, b1 = obj
    l2, t2, r2, b2 = area
    obj_area = (r1 - l1) * (b1 - t1)
    area_area = (r2 - l2) * (b2 - t2)
    l = max(l1, l2)
    t = max(t1, t2)
    r = min(r1, r2)
    b = min(b1, b2)
    w = r - l
    h = b - t
    if w <= 0 or h <= 0:
        return 0
    its = w * h
    fgd = obj_area + area_area - its
    assert fgd > 0
    return its / fgd


def inside_confusion_areas(obj, confusion_areas):
    iou = [calc_iou(obj, area) for area in confusion_areas]
    return max(iou) > 0.5 if iou else False


def parse_groud_truth(ground_truth_file, class_idx):
    ground_truth = defaultdict(list)
    img_paths = {}
    num_objs = 0
    with open(ground_truth_file) as f:
        lines = [line for line in f.read().splitlines() if line]
    for line in tqdm(lines, desc='Loading ground truth ...'):
        image_path = line
        key = os.path.basename(image_path)
        img_paths[key] = image_path
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        with open(label_path) as f:
            objs = [obj for obj in f.read().splitlines() if obj]
        for obj in objs:
            c, xc, yc, w, h = obj.split()
            c = int(c)
            if class_idx >= 0 and class_idx != c:
                continue
            xc, yc, w, h = [float(v) for v in [xc, yc, w, h]]
            l = xc - 0.5 * w
            r = xc + 0.5 * w
            t = yc - 0.5 * h
            b = yc + 0.5 * h

            ground_truth[key].append([l, t, r, b])
            num_objs += 1
    print('Loaded %d objects in %d images' % (num_objs, len(ground_truth)))
    return ground_truth, img_paths


def parse_detect_result(detect_result_dir, ground_truth, class_idx):
    raw_detect_result = defaultdict(list)
    num_objs = 0
    label_paths = glob(os.path.join(detect_result_dir, '*.txt'))
    for label_path in tqdm(label_paths, desc='Loading detect result ...'):
        key = os.path.basename(label_path).replace('.txt', '.jpg')
        # assert key in ground_truth
        with open(label_path) as f:
            objs = [obj for obj in f.read().splitlines() if obj]
        for obj in objs:
            c, xc, yc, w, h = obj.split()
            c = int(c)
            if class_idx >= 0 and class_idx != c:
                continue
            xc, yc, w, h = [float(v) for v in [xc, yc, w, h]]
            l = max(xc - 0.5 * w, 0)
            r = min(xc + 0.5 * w, 1)
            t = max(yc - 0.5 * h, 0)
            b = min(yc + 0.5 * h, 1)
            if l >= r or t >= b:
                continue
            raw_detect_result[key].append([l, t, r, b])
            num_objs += 1
    print('Loaded %d objects in %d images' % (num_objs, len(ground_truth)))
    return raw_detect_result


def parse_confusion_area(confusion_area_file):
    print('Loading confusion areas from %s ...' % confusion_area_file)
    with open(confusion_area_file) as f:
        objs = json.load(f)
    confusion_areas = {}
    num_areas = 0
    for img_path in tqdm(objs):
        key = os.path.basename(img_path)
        assert key not in confusion_areas
        confusion_areas[key] = []
        for l, t, r, b, c in objs[img_path]:
            assert c == 1047936
            confusion_areas[key].append([l, t, r, b])
            num_areas += 1

    print('Loaded %d confusion areas in %d images' % (num_areas, len(confusion_areas)))
    return confusion_areas


def filter_by_iou(objs, iou_thres):
    valid_objs = []
    objs.sort(key=lambda x: x[-1], reverse=True)
    num_objs = len(objs)
    removed = [False] * num_objs
    for i in range(num_objs):
        if removed[i]:
            continue
        valid_objs.append(objs[i])
        for j in range(i + 1, num_objs):
            if max(calc_iou(objs[i][:4], objs[j][:4]), calc_iou(objs[j][:4], objs[i][:4])) > iou_thres:
                removed[j] = True
    return valid_objs


def filter_detect_result(raw_detect_result, conf_thres, iou_thres):
    detect_result = defaultdict(list)
    for img_path in tqdm(raw_detect_result, 'Filtering detection result ...'):
        for l, t, r, b, s in raw_detect_result[img_path]:
            if s > conf_thres:
                detect_result[img_path].append([l, t, r, b, s])
        if img_path in detect_result and iou_thres > 0:
            detect_result[img_path] = filter_by_iou(detect_result[img_path], iou_thres)
    return detect_result


def calc_metrics(gts, dets, iou_thres):
    if not gts or not dets:
        return [], [], dets, gts

    tps, ious_per_img, fps, fns = [], [], [], []
    matched = [False] * len(dets)
    dets = np.float32(dets)
    for gl, gt, gr, gb in gts:
        l = np.maximum(dets[:, 0], gl)
        t = np.maximum(dets[:, 1], gt)
        r = np.minimum(dets[:, 2], gr)
        b = np.minimum(dets[:, 3], gb)
        w = np.maximum(r - l, 0)
        h = np.maximum(b - t, 0)
        intersect = w * h
        union = (
                (gr - gl) * (gb - gt)
                + (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                - intersect
        )
        ious = intersect / union
        max_iou = np.max(ious)
        max_iou_idx = np.argmax(ious)
        if matched[max_iou_idx]:
            fns.append([gl, gt, gr, gb])
        else:
            if max_iou > iou_thres:
                l, t, r, b = dets[max_iou_idx]
                tps.append([l, t, r, b])
                ious_per_img.append(max_iou)
                matched[max_iou_idx] = True
            else:
                fns.append([gl, gt, gr, gb])
    for (l, t, r, b), m in zip(dets, matched):
        if not m:
            fps.append([l, t, r, b])

    return tps, ious_per_img, fps, fns


def visualize(img_path, tps, fps, fns, vis_dir, display_score, channels):
    assert channels in [3, 6]

    if not fps and not fns:
        return

    img = cv.imread(img_path, cv.IMREAD_COLOR)
    assert img is not None, img_path
    h, w = img.shape[:2]
    if channels == 6:
        w = w // 2
        imgs = [img[:, :w], img[:, w:]]

    def cvt_coords(l, t, r, b, w, h):
        l, r = l * w, r * w
        t, b = t * h, b * h
        return [int(v) for v in [l, t, r, b]]

    # Plot true positive
    color = (0, 255, 0)
    for l, t, r, b in tps:
        l, t, r, b = cvt_coords(l, t, r, b, w, h)
        if channels == 3:
            cv.rectangle(img, (l, t), (r, b), color, 2)
        else:
            for i in range(2):
                cv.rectangle(imgs[i], (l, t), (r, b), color, 2)

    # Plot false positive
    color = (0, 0, 255)
    for l, t, r, b in fps:
        l, t, r, b = cvt_coords(l, t, r, b, w, h)
        if channels == 3:
            cv.rectangle(img, (l, t), (r, b), color, 2)
        else:
            for i in range(2):
                cv.rectangle(imgs[i], (l, t), (r, b), color, 2)

    # Plot false negative
    color = (255, 0, 0)
    for l, t, r, b in fns:
        l, t, r, b = cvt_coords(l, t, r, b, w, h)
        if channels == 3:
            cv.rectangle(img, (l, t), (r, b), color, 2)
        else:
            for i in range(2):
                cv.rectangle(imgs[i], (l, t), (r, b), color, 2)

    vis_path = os.path.join(vis_dir, os.path.basename(img_path))
    if channels == 3:
        assert cv.imwrite(vis_path, img)
    else:
        assert cv.imwrite(vis_path, np.hstack(imgs))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_file', default='', type=str)
    parser.add_argument('--detect_result_dir', default='', type=str)
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--summary_file', type=str, default='/home/jiayu/yolov5/runs/eval/exp001/performance_report.json')
    parser.add_argument('--vis_dir', type=str, default='/home/jiayu/yolov5/runs/eval/exp001/vis')
    parser.add_argument('--vis_url_dir', type=str, default='/home/jiayu/yolov5/runs/eval/exp001/vis')
    parser.add_argument('--remote_vis_dir', type=str, default='http:')
    parser.add_argument('--hide_score', action='store_true')
    parser.add_argument('--class_idx', type=int, default=-1)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--output_infer_wrong_dir', type=str, default='/home/jiayu/yolov5/runs/eval/exp001/output.csv')

    opt = parser.parse_args()
    print(opt)

    return opt

def moveFileToDest(ground_truth):
    root = ground_truth + "/labels"
    dest = ground_truth
    fileList = []
    if os.path.exists(root):
        for item in os.listdir(root):
            filepath = os.path.join(root,item)
            if os.path.isdir(filepath):
                list = moveFileToDest(filepath,dest)
                fileList = fileList + list
            elif os.path.isfile(filepath):
                shutil.move(filepath,dest)
        #return fileList
        os.rmdir(root)

def run(
        ground_truth_file='data/converted_data/train/valid.list',
        detect_result_dir='detect_with_onnx',
        conf_thres=0.25,
        iou_thres=0.45,
        summary_file='output/performance_report.json',
        vis_dir='output/vis',
        vis_url_dir='',
        remote_vis_dir='http:',
        hide_score=True,
        class_idx=-1,
        channels=3,
        output_infer_wrong_dir='output/output.csv',
):
    print(ground_truth_file)

    if conf_thres > 0:
        conf_thres = [conf_thres]
    else:
        conf_thres = [0.1 * v for v in range(1, 10)]

    if iou_thres > 0:
        iou_thres = [iou_thres]
    else:
        iou_thres = [0.1 * v for v in range(5, 10)]
    assert len(iou_thres) == 1 and len(conf_thres) == 1

    # Move to one dir
    moveFileToDest(detect_result_dir)
    
    summary_dir = summary_file.split('performance_report.json', 1)[0]
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(vis_url_dir, exist_ok=True)

    summary = {}
    summary_keys = ['precision', 'recall', 'f1_score', 'img_num', 'gt_num', 'det_num', 'tp', 'fp', 'fn']
    summary_txt = [summary_keys]

    ## Load ground truth and detection result

    ground_truth, img_paths = parse_groud_truth(ground_truth_file, class_idx)
    raw_detect_result = parse_detect_result(detect_result_dir, ground_truth, class_idx)

    ## Init dirs
    if vis_dir:
        shutil.rmtree(vis_dir, ignore_errors=True)
        os.makedirs(vis_dir, exist_ok=True)

    ## Evaluation
    num_img = len(open(ground_truth_file, 'r').readlines())
    for it in iou_thres:
        for st in conf_thres:

            xmin = []
            ymin = []
            xmax = []
            ymax = []
            imgurl = []

            num_gt, num_det = 0, 0
            num_tp, num_fp, num_fn = 0, 0, 0
            ious = []
            #detect_result = filter_detect_result(raw_detect_result, st, it)
            for key in tqdm(ground_truth, 'IoU thres = %.2f and score thres = %.2f ...' % (it, st)):
                tps, ious_per_img, fps, fns = calc_metrics(ground_truth[key], raw_detect_result[key], it)
                assert len(tps) + len(fps) == len(raw_detect_result[key]) and len(tps) + len(fns) == len(
                    ground_truth[key])
                assert len(tps) == len(ious_per_img)
                num_gt += len(ground_truth[key])
                num_det += len(raw_detect_result[key])
                num_tp += len(tps)
                num_fp += len(fps)
                num_fn += len(fns)
                ious.extend(ious_per_img)

                if vis_dir:
                    visualize(img_paths[key], tps, fps, fns, vis_dir, not hide_score, channels)

                # output the ML recognized error image
                if fps or fns:
                    for labels in [tps, fps]:
                        for l, t, r, b in labels:
                            xmin.append(l)
                            ymin.append(t)
                            xmax.append(r)
                            ymax.append(b)
                            imgurl.append('https://fileman.clobotics.cn/api/file/' + key[:-4])

            productid = [1] * len(xmin)
            df = pd.DataFrame(
                {'ImgUrl': imgurl, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'ProductId': productid})

            os.makedirs(output_infer_wrong_dir, exist_ok=True)
            df.to_csv(output_infer_wrong_dir + "/output.csv", index=False)

            prec = num_tp / num_det if num_det > 0 else 0
            rec = num_tp / num_gt if num_gt > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec + 1e-16)
            summary_values = ['%.3f' % prec, '%.3f' % rec, '%.3f' % f1, num_img, num_gt, num_det, num_tp, num_fp,
                                num_fn]
            summary = dict(zip(summary_keys, summary_values))
            summary_txt.append(summary_values)

    performance_report_info = {}
    data = json.loads(json.dumps(performance_report_info))

    stage = os.path.basename(detect_result_dir)

    if stage == 'previous_val':
        postfix = 'previous_eval'
    elif stage == 'increment_val':
        postfix = 'increment_eval'
    else:
        postfix = 'training_res'
    data[postfix] = summary

    with open(summary_file, 'w+') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
    print('Saved summary to %s' % summary_file)

    if os.path.exists(vis_dir):
        vis_list = os.listdir(vis_dir)
        if vis_url_dir:
            blob_vis_list = [remote_vis_dir + vis for vis in vis_list]
            writer = pd.DataFrame(columns=['Imgurl'], data=blob_vis_list)
            writer.to_csv(vis_url_dir + '/vis_url.csv', encoding="UTF-8", index=False)


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)