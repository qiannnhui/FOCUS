import os
import cv2
import glob

# 原始資料夾路徑
image_dir = "new09_DS/images"
mask_dir = "new09_DS/ground_truth_mask"

# 輸出資料夾路徑
output_image_dir = "resized_folder/images"
output_mask_dir = "resized_folder/ground_truth_mask"

# 確保輸出資料夾存在
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 取得所有圖片和對應的 Mask 路徑
image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

for image_path, mask_path in zip(image_files, mask_files):
    # 讀取圖片和 Mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # 取得圖片大小
    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape[:2]

    # 計算最大尺寸
    target_height = max(h_img, h_mask)
    target_width = max(w_img, w_mask)

    # 進行 Resize，使用雙線性插值
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # 取得檔名並儲存
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_image_dir, filename), resized_image)
    cv2.imwrite(os.path.join(output_mask_dir, filename.replace(".jpg", ".png")), resized_mask)

    print(f"Processed: {filename}")
