'''
Descripttion: split_img.py
version: 1.0
Author: UniDome
Date: 2022-04-20 16:28:45
LastEditors: UniDome
LastEditTime: 2022-04-20 16:39:56
'''
import os, shutil, random
from tqdm import tqdm

def split_img(img_path, label_path, split_list):
    # 创建数据集文件夹

    Data = '/data/disk3/cv2022/data/posm_train_v6_sp'
    train_img_dir = Data + '/images/train'
    val_img_dir = Data + '/images/val'
    test_img_dir = Data + '/images/test'

    train_label_dir = Data + '/labels/train'
    val_label_dir = Data + '/labels/val'
    test_label_dir = Data + '/labels/test'

    try:
        # 创建文件夹
        os.mkdir(Data)
        os.makedirs(train_img_dir)
        os.makedirs(train_label_dir)
        os.makedirs(val_img_dir)
        os.makedirs(val_label_dir)
        os.makedirs(test_img_dir)
        os.makedirs(test_label_dir)
    except:
        print("文件目录已存在")

    train, val, test = split_list
    all_img = os.listdir(img_path)
    all_img_path = [os.path.join(img_path, img) for img in all_img]
    # all_label = os.listdir(label_path)
    # all_label_path = [os.path.join(label_path, label) for label in all_label]
    train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    train_img_copy = [os.path.join(train_img_dir, img.split('\\')[-1]) for img in train_img]
    train_label = [toLabelPath(img, label_path) for img in train_img]
    train_label_copy = [os.path.join(train_label_dir, label.split('\\')[-1]) for label in train_label]
    for i in tqdm(range(len(train_img)), desc='train ', ncols=80, unit='img'):
        try:
            _copy(train_label[i], train_label_dir)
            _copy(train_img[i], train_img_dir)
            all_img_path.remove(train_img[i])
        except:
            all_img_path.remove(train_img[i])
    val_img = random.sample(all_img_path, int(val / (val + test) * len(all_img_path)))
    val_label = [toLabelPath(img, label_path) for img in val_img]
    for i in tqdm(range(len(val_img)), desc='val ', ncols=80, unit='img'):
        try:
            _copy(val_label[i], val_label_dir)
            _copy(val_img[i], val_img_dir)
        except:
            all_img_path.remove(val_img[i])
    test_img = all_img_path
    test_label = [toLabelPath(img, label_path) for img in test_img]
    for i in tqdm(range(len(test_img)), desc='test ', ncols=80, unit='img'):
        try:
            _copy(test_label[i], test_label_dir)
            _copy(test_img[i], test_img_dir)
        except:
            continue


def _copy(from_path, to_path):
    shutil.copy(from_path, to_path)

def toLabelPath(img_path, label_path):
    img = img_path.split('/')[-1]
    label = img.split('.jpg')[0] + '.txt'
    return os.path.join(label_path, label)

def main():
    # 图片与标注路径
    img_path   = '/data/disk3/cv2022/data/posm_train_v6/dataset/images'
    label_path = '/data/disk3/cv2022/data/posm_train_v6/dataset/labels'
    split_list = [0.9, 0.05, 0.05]	# 数据集划分比例[train:val:test]
    split_img(img_path, label_path, split_list)

if __name__ == '__main__':
    main()