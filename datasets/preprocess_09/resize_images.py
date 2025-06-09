import os
import cv2
import glob
import argparse

def resize_images_and_masks(source_root: str):
    """
    Resize images and masks to the maximum height and width of each pair.
    The resized images and masks will be saved in a new folder structure.
    """
    image_dir = os.path.join(source_root, "images")
    mask_dir = os.path.join(source_root, "ground_truth_mask")

    output_image_dir = image_dir
    output_mask_dir = mask_dir

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    for image_path, mask_path in zip(image_files, mask_files):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape[:2]

        target_height = max(h_img, h_mask)
        target_width = max(w_img, w_mask)

        # Resize images and masks to the target size
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_image_dir, filename), resized_image)
        cv2.imwrite(os.path.join(output_mask_dir, filename.replace(".jpg", ".png")), resized_mask)

        # print(f"Processed: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images and masks to the maximum height and width of each pair.")
    parser.add_argument("--source_root", type=str, default="09_val", help="Path to the source dataset root.")
    args = parser.parse_args()
    resize_images_and_masks(source_root=args.source_root)
    print("Resizing completed.")