import os
import shutil
from glob import glob
import argparse

# The original folder is in the format for anomaly detection, use these functions to convert it to the format for semantic segmentation.
def convert_to_semantic_segmentation_format(source_root: str, target_root=None):
    """
    Convert the dataset from anomaly detection format to semantic segmentation format.
    The original dataset is in the format:
    - 09/
        - folder1/
            - ground_truth/
                - 09/
                    - test/
                        - 000_mask.png
            - test/
                - 09/
                    - 000.jpg
        - folder2/
            ...
    The new dataset will be in the format:
    - new09_DS/
        - images/
            - folder1.jpg
            - folder2.jpg
            ...
        - ground_truth_mask/
            - folder1.png
            - folder2.png
            ...
    """

    source_root = source_root
    target_root = target_root if target_root else os.path.join(os.path.dirname(source_root), 'new_DS')
    image_dest = os.path.join(target_root, 'images')
    mask_dest = os.path.join(target_root, 'ground_truth_mask')

    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(mask_dest, exist_ok=True)

    for folder in glob(os.path.join(source_root, '*')):
        if os.path.isdir(folder):
            folder_name = os.path.basename(folder)

            mask_path = os.path.join(folder, 'ground_truth/09/test/000_mask.png')
            image_path = os.path.join(folder, 'test/09/000.jpg')

            if os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(mask_dest, f"{folder_name}.png"))
            else:
                print(f"Cannot find: {mask_path}")

            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(image_dest, f"{folder_name}.jpg"))
            else:
                print(f"Cannot find: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset from anomaly detection format to semantic segmentation format.")
    parser.add_argument('--dataset', type=str, default="09", help='Dataset name to convert. Default is "09".')
    parser.add_argument('--source_root', type=str, default="/home/FOCUS/datasets", help='Path to the source dataset root directory.')
    parser.add_argument('--target_root', type=str, default="/home/FOCUS/datasets", help='Path to the new dataset folder. If not provided, it will be created in the same directory as source_root.')

    args = parser.parse_args()
    source_root = os.path.join(args.source_root, args.dataset)
    target_root = os.path.join(args.target_root, f"{args.dataset}_val")
    
    convert_to_semantic_segmentation_format(source_root, target_root)
    