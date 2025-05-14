import os
import shutil
from glob import glob

# 原始資料夾路徑
source_root = '09'  # 根據你的範例圖，這是資料夾名稱
# 新的資料夾路徑
new_folder = 'new09_DS'
image_dest = os.path.join(new_folder, 'images')
mask_dest = os.path.join(new_folder, 'ground_truth_mask')

# 確認目標資料夾存在，不存在則建立
os.makedirs(image_dest, exist_ok=True)
os.makedirs(mask_dest, exist_ok=True)

# 遍歷所有09底下的子資料夾
for folder in glob(os.path.join(source_root, '*')):
    if os.path.isdir(folder):
        folder_name = os.path.basename(folder)

        # 定位ground truth和image的路徑
        mask_path = os.path.join(folder, 'ground_truth/09/test/000_mask.png')
        image_path = os.path.join(folder, 'test/09/000.jpg')

        # 確認檔案存在才進行複製
        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(mask_dest, f"{folder_name}.png"))
            # print(f"複製: {mask_path} -> {mask_dest}/{folder_name}.png")
        else:
            print(f"找不到: {mask_path}")

        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(image_dest, f"{folder_name}.jpg"))
            # print(f"複製: {image_path} -> {image_dest}/{folder_name}.jpg")
        else:
            print(f"找不到: {image_path}")
