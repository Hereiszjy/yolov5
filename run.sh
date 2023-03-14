# inference
python detect.py --weights /home/jiayu/yolov5/runs/train/ccth_Fanta_11/weights/best.pt \
--data /data/disk3/cv2022/data/ccth_Fanta_11/dataset/dataset.yaml --source /data/disk3/cv2022/data/ccth_Fanta_11/dataset/images \
--img-size 960 --save-txt --nosave

# train
python train.py --weights /home/jiayu/yolov5/yolov5l.pt --img-size 960 --device 5 --batch-size 64 \
--epochs 100 --data /home/jiayu/yolov5/data/posm_new.yaml --name posm_0105 --cos-lr --image-weights

# eval
python val.py --data /data/disk3/cv2022/data/ccth_Fanta_11/dataset/dataset.yaml --weights runs/train/ccth_Fanta_11/weights/best.pt \
--conf-thres 0.5 --task val --device 5 --name ccth_Fanta_11 --img-size 960

# 训练jjcn的posm
python train.py --weights /home/jiayu/yolov5/yolov5l.pt --img-size 960 --device 5 --batch-size 16 --epochs 50 \
--data /home/jiayu/yolov5/data/posm_new.yaml --name posm_0112v2 --cos-lr --image-weights --resume

# export
python export.py --weights runs/train/sbd_20230216/weights/best.pt --include onnx --img 960

# evaluate
python evaluate.py --ground_truth_file /data/disk3/cv2022/data/ccth_Fanta_11/dataset/val.list \
--detect_result_dir /home/jiayu/yolov5/runs/detect/exp7/labels 