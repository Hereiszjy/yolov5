from pathlib import Path
import pandas as pd

df = pd.read_csv("/home/jiayu/yolov5/sku_list/jjcn_posm_230417.csv")
sku_id = df["ProductId"].values.tolist()
sku_id.sort()
tmp = []
for index, sku in enumerate(sku_id):
    tmp.append(f"{sku}, {index}")
label_txt = Path("/home/jiayu/yolov5/sku_list/label.txt")
label_txt.write_text('\n'.join(tmp))