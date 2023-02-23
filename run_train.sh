# ccth_10
DATASET=$1
# posm, unit
TASK=$2
# export CUDA_VISIBLE_DEVICES=5

# dwomload img 
python scripts/download_images.py --work_dir /data/disk3/cv2022/data/$DATASET
# generate dataset for yolov5
python scripts/dataset.py --root_dir /data/disk3/cv2022/data/$DATASET --skulist_dir /home/jiayu/yolov5/sku_list/$DATASET.csv

# train
if [ $TASK = unit ]
then
    python train.py --img-size 960 --device 5 --batch-size 64 --epochs 100 \
    --data /data/disk3/cv2022/data/$DATASET/dataset/dataset.yaml --name $DATASET --image-weight 
elif [ $TASK = posm ]
then 
    python train.py --weights /home/jiayu/yolov5/yolov5l.pt --img-size 960 --device 3,4,5 --batch-size 48 --epochs 50 \
    --data /data/disk3/cv2022/data/$DATASET/dataset/dataset.yaml --name $DATASET --cos-lr --image-weight
fi