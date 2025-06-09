# The original folder is in the format for anomaly detection, use these functions to convert it to the format for semantic segmentation.
python datasets/preprocess_09/change_datasets_format.py --dataset 09
python datasets/preprocess_09/resize_images.py --source_root datasets/09_val
python datasets/preprocess_09/multi_mask_to_single.py --folder_path datasets/09_val/ground_truth_mask
python utils/prepare/prepare_09_val.py