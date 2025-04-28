import glob
import json
import os
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
import copy
from tqdm import tqdm

def create_semantic_dataset(category_dir, dataset_type, category_id, category_name):
    """
    Convert dataset into COCO Semantic Segmentation format.
    """
    images = []
    annotations = []
    ann_id = 1

    # 遍歷所有 PNG 圖像
    image_files = sorted(glob.glob(os.path.join(category_dir, dataset_type, "**", "*.png"), recursive=True))

    for img_id, image_file in tqdm(enumerate(image_files, start=1), total=len(image_files)):
        relative_path = os.path.relpath(image_file, category_dir)
        file_name = os.path.join(category_name, relative_path)
        image = Image.open(image_file)
        width, height = image.size

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # 處理標註
        # print("image_file = ", image_file)
        if dataset_type == "test":
            gt_path = copy.deepcopy(image_file)
            gt_path = gt_path.replace("test", "ground_truth")
            mask_file = gt_path.replace(".png", "_mask.png")
            gt_file_name = copy.deepcopy(file_name)
            gt_file_name = gt_file_name.replace("test", "ground_truth")
            gt_file_name = gt_file_name.replace(".png", "_mask.png")
            # print("mask_filename = ", mask_file)
            # print(os.path.exists(mask_file))
            if not os.path.exists(mask_file):
                continue  # 無標註則跳過
        elif dataset_type == "train":
            continue

        mask = np.array(Image.open(mask_file).convert("L"))
        foreground_mask = (mask > 128).astype(np.uint8)
        background_mask = (mask == 0).astype(np.uint8)

        fg_rle = mask_util.encode(np.asfortranarray(foreground_mask))
        fg_rle['counts'] = fg_rle['counts'].decode('utf-8')

        bg_rle = mask_util.encode(np.asfortranarray(background_mask))
        bg_rle['counts'] = bg_rle['counts'].decode('utf-8')

        fg_area = mask_util.area(fg_rle).tolist()
        bg_area = mask_util.area(bg_rle).tolist()

        fg_bbox = mask_util.toBbox(fg_rle).tolist()
        bg_bbox = mask_util.toBbox(bg_rle).tolist()

        annotations.append({
            "id": ann_id,
            "file_name": gt_file_name,
            "image_id": img_id,
            "category_id": category_id,
            "segmentation": fg_rle,
            "area": fg_area,
            "bbox": fg_bbox,
            "iscrowd": 0
        })

        ann_id += 1

    return images, annotations

def create_dataset(dataset_root, output_dir):
    train_images = []
    train_annotations = []
    test_images = []
    test_annotations = []
    all_categories = []

    ann_id = 1
    category_id = 1

    categories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

    for category in categories:
        category_dir = os.path.join(dataset_root, category)

        # 處理 train 資料
        train_path = os.path.join(category_dir, "train")
        if os.path.exists(train_path):
            category_images, category_annotations = create_semantic_dataset(category_dir, "train", category_id, category)
            train_images.extend(category_images)
            train_annotations.extend(category_annotations)

        # 處理 test 資料
        test_path = os.path.join(category_dir, "test")
        if os.path.exists(test_path):
            category_images, category_annotations = create_semantic_dataset(category_dir, "test", category_id, category)
            test_images.extend(category_images)
            test_annotations.extend(category_annotations)

        all_categories.append({"id": category_id, "name": category, "supercategory": category, "is_thing": 1})
        category_id += 1

    # 儲存 train 資料
    coco_train_format = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": all_categories
    }
    os.makedirs(output_dir, exist_ok=True)
    output_train_file = os.path.join(output_dir, "mvtec2_train_SEMANTIC.json")
    with open(output_train_file, 'w') as f:
        json.dump(coco_train_format, f)
    print(f"Saved Train Semantic Segmentation JSON: {output_train_file}")

    # 儲存 test 資料
    coco_test_format = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": all_categories
    }
    output_test_file = os.path.join(output_dir, "mvtec2_val_SEMANTIC.json")
    with open(output_test_file, 'w') as f:
        json.dump(coco_test_format, f)
    print(f"Saved Test Semantic Segmentation JSON: {output_test_file}")

if __name__ == "__main__":
    dataset_root = "datasets/MVTEC2"
    output_root = "datasets/MVTEC2"

    # 呼叫函數來產生 train 和 test 資料的 JSON 檔案
    create_dataset(dataset_root, output_root)



# import glob
# import json
# import os
# from PIL import Image
# import numpy as np
# import pycocotools.mask as mask_util
# import tqdm
# from pathlib import Path

# def create_instance_dataset(category_dir, category_name, dataset_type, output_dir):
#     """
#     Convert dataset into COCO Instance Segmentation format.

#     - Suitable for Mask R-CNN.
#     - Objects are individually labeled.
#     """
#     images = []
#     annotations = []
#     ann_id = 1

#     # 遞迴尋找所有子資料夾內的圖片
#     image_files = sorted(glob.glob(os.path.join(category_dir, "**", "*.png"), recursive=True))

#     for img_id, image_file in tqdm.tqdm(enumerate(image_files, start=1), total=len(image_files)):
#         file_name = os.path.relpath(image_file, category_dir)  # 保留子資料夾結構
#         image = Image.open(image_file)
#         width, height = image.size

#         images.append({
#             "id": img_id,
#             "file_name": file_name,
#             "width": width,
#             "height": height
#         })

#         # 只有 `test` 有標註
#         if dataset_type == "test":
#             mask_dir = os.path.join(os.path.dirname(category_dir), "ground_truth")
#             mask_file = os.path.join(mask_dir, file_name)

#             if not os.path.exists(mask_file):
#                 continue  # 無標註則跳過

#             # 讀取 mask
#             mask = np.array(Image.open(mask_file).convert("L"))
#             binary_mask = (mask > 128).astype(np.uint8)
#             rle = mask_util.encode(np.asfortranarray(binary_mask))
#             rle['counts'] = rle['counts'].decode('utf-8')

#             bbox = mask_util.toBbox(rle).tolist()
#             area = mask_util.area(rle).tolist()

#             annotations.append({
#                 "id": ann_id,
#                 "image_id": img_id,
#                 "category_id": 1,
#                 "segmentation": rle,
#                 "area": area,
#                 "bbox": bbox,
#                 "iscrowd": 0
#             })
#             ann_id += 1

#     categories = [{"id": 1, "name": "foreground", "supercategory": "foreground"}]
#     coco_format = {"images": images, "annotations": annotations, "categories": categories}

#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, f"{category_name}_{dataset_type}_INSTANCE.json")

#     with open(output_file, 'w') as f:
#         json.dump(coco_format, f)
#     print(f"Saved Instance Segmentation JSON: {output_file}")

# def create_semantic_dataset(category_dir, category_name, dataset_type, output_dir):
#     """
#     Convert dataset into COCO Semantic Segmentation format.

#     - Suitable for Panoptic Segmentation.
#     - Foreground and background are treated as separate classes.
#     """
#     images = []
#     annotations = []
#     ann_id = 1

#     # 遞迴尋找所有子資料夾內的圖片
#     image_files = sorted(glob.glob(os.path.join(category_dir, "**", "*.png"), recursive=True))

#     for img_id, image_file in tqdm.tqdm(enumerate(image_files, start=1), total=len(image_files)):
#         file_name = os.path.relpath(image_file, category_dir)  # 保留子資料夾結構
#         image = Image.open(image_file)
#         width, height = image.size

#         images.append({
#             "id": img_id,
#             "file_name": file_name,
#             "width": width,
#             "height": height
#         })

#         # 只有 `test` 有標註
#         if dataset_type == "test":
#             mask_dir = os.path.join(os.path.dirname(category_dir), "ground_truth")
#             mask_file = os.path.join(mask_dir, file_name)

#             if not os.path.exists(mask_file):
#                 continue  # 無標註則跳過

#             mask = np.array(Image.open(mask_file).convert("L"))
#             foreground_mask = (mask > 128).astype(np.uint8)
#             background_mask = (mask == 0).astype(np.uint8)

#             fg_rle = mask_util.encode(np.asfortranarray(foreground_mask))
#             fg_rle['counts'] = fg_rle['counts'].decode('utf-8')

#             bg_rle = mask_util.encode(np.asfortranarray(background_mask))
#             bg_rle['counts'] = bg_rle['counts'].decode('utf-8')

#             fg_area = mask_util.area(fg_rle).tolist()
#             bg_area = mask_util.area(bg_rle).tolist()

#             fg_bbox = mask_util.toBbox(fg_rle).tolist()
#             bg_bbox = mask_util.toBbox(bg_rle).tolist()

#             annotations.append({
#                 "image_id": img_id,
#                 "file_name": file_name,
#                 "segments_info": [
#                     {"id": ann_id, "category_id": 1, "area": fg_area, "bbox": fg_bbox, "iscrowd": 0},
#                     {"id": ann_id + 1, "category_id": 2, "area": bg_area, "bbox": bg_bbox, "iscrowd": 0}
#                 ]
#             })
#             ann_id += 2

#     categories = [
#         {"id": 1, "name": "foreground", "supercategory": "foreground", "is_thing": 1},
#         {"id": 2, "name": "background", "supercategory": "background", "is_thing": 0}
#     ]

#     coco_format = {"images": images, "annotations": annotations, "categories": categories}

#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, f"{category_name}_{dataset_type}_SEMANTIC.json")

#     with open(output_file, 'w') as f:
#         json.dump(coco_format, f)
#     print(f"Saved Semantic Segmentation JSON: {output_file}")


# def convert_mask(mask_file):
#     """Convert a mask to Detectron2-compatible format (shift pixel values)."""
#     if not os.path.exists(mask_file):
#         print(f"Warning: Mask file {mask_file} not found!")
#         return None

#     mask = np.array(Image.open(mask_file).convert("L"))
#     mask = mask + 1  # Shift all pixel values by 1
#     return mask


# if __name__ == "__main__":
#     dataset_root = "datasets/MVTEC"
#     output_root = "datasets/MVTEC"

#     categories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

#     for category in categories:
#         category_dir = os.path.join(dataset_root, category)
#         for dataset_type in ["train", "test"]:
#             dataset_path = os.path.join(category_dir, dataset_type)
#             if os.path.exists(dataset_path):
#                 create_instance_dataset(dataset_path, category, dataset_type, output_root)
#                 create_semantic_dataset(dataset_path, category, dataset_type, output_root)

#         # Convert masks for Detectron2
#         mask_dir = os.path.join(category_dir, "ground_truth")
#         converted_mask_dir = os.path.join(category_dir, "annotations_detectron2")
#         os.makedirs(converted_mask_dir, exist_ok=True)

#         for mask_file in tqdm.tqdm(glob.glob(os.path.join(mask_dir, "*.png"))):
#             converted_mask = convert_mask(mask_file)
#             if converted_mask is not None:
#                 Image.fromarray(converted_mask).save(os.path.join(converted_mask_dir, os.path.basename(mask_file)))

