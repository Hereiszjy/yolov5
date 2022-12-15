1. 下载图片
python download_images.py --work_dir /data/disk2/cv2022/unit_data/posm

2. 生成标注文件
python labelme2yolov5.py /data/disk2/cv2022/unit_data/posm

3. 拆分数据集
python split_img.py