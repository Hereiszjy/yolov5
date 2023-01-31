# inference
python detect.py --weights /home/jiayu/yolov5/runs/train/posm_0110/weights/best.pt \
--data /home/jiayu/yolov5/data/posm_test.yaml --source /data/disk3/cv2022/data/posm_test_gt/dataset/images

# train
python train.py --weights /home/jiayu/yolov5/yolov5l.pt --img-size 960 --device 5 --batch-size 64 \
--epochs 100 --data /home/jiayu/yolov5/data/posm_new.yaml --name posm_0105 --cos-lr --image-weights

# eval
python val.py --data data/sbd_test.yaml --weights runs/train/sbd2/weights/best.pt \
--conf-thres 0.5 --task test --device 5 --name sbd_0130_0.5

# 训练jjcn的posm
python train.py --weights /home/jiayu/yolov5/yolov5l.pt --img-size 960 --device 5 --batch-size 16 --epochs 50 \
--data /home/jiayu/yolov5/data/posm_new.yaml --name posm_0112v2 --cos-lr --image-weights --resume

# export
python export.py --weights /home/jiayu/yolov5/runs/train/posm_0105/weights/best.pt --include onnx --img 960