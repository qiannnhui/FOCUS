import os
import cv2
import numpy as np

def convert_non_black_to_white(img):
    # 假設圖像是彩色的 (BGR格式)
    # 將圖像中所有非黑色像素設定為白色
    lower_black = np.array([0, 0, 0])  # 黑色的最小範圍
    upper_black = np.array([50, 50, 50])  # 黑色的最大範圍
    
    # 創建黑色區域的掩模
    black_mask = cv2.inRange(img, lower_black, upper_black)
    
    # 將不是黑色的部分設為白色
    img[black_mask == 0] = [255, 255, 255]  # 這裡使用白色替換非黑色區域
    
    return img

def process_images_in_folder(folder_path):
    # 遍歷資料夾中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 只處理圖片檔案
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 讀取圖片
            img = cv2.imread(file_path)
            if img is not None:
                # 處理圖片
                result = convert_non_black_to_white(img)
                
                # 保存處理後的圖片
                output_path = os.path.join(folder_path, f"{filename}")
                cv2.imwrite(output_path, result)
                print(f"Processed: {filename}")
            else:
                print(f"Failed to read: {filename}")
                
# 使用範例
folder_path = "new09_DS/ground_truth_mask"
process_images_in_folder(folder_path)
