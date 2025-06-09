import cv2
import glob

image_dir = "09_val/images"
mask_dir = "09_val/ground_truth_mask"

image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
mask_files = sorted(glob.glob(f"{mask_dir}/*.png"))

for img_path, mask_path in zip(image_files, mask_files):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    
    if image.shape[:2] != mask.shape[:2]:
        print(f"Dimension mismatch: {img_path} vs {mask_path}")