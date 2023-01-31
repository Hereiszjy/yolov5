1. 下载图片
python download_images.py --work_dir /data/disk3/cv2022/data/posm_test_gt_v2

2. 生成标注文件
python labelme2yolov5.py /data/disk3/cv2022/data/posm_test_gt_v2

3. 拆分数据集
python split_img.py