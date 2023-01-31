export DATASET=sbd

python scritps/download_images.py --work_dir /data/disk3/cv2022/data/$DATASET
python scritps/labelme2yolov5.py /data/disk3/cv2022/data/$DATASET

# # Train
python train.py --weights /home/jiayu/yolov5/yolov5l.pt --img-size 960 --device 5 --batch-size 16 --epochs 50 \
--data /home/jiayu/yolov5/data/sbd.yaml --name sbd --cos-lr --image-weight