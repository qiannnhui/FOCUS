import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

MVTEC_CATEGORIES = [
    {"color": [255, 255, 255], "id": 1, "isthing": 1, "name": "foreground"},
    {"color": [0, 0, 0], "id": 2, "isthing": 0, "name": "background"},
]

MVTEC_COLORS = [k["color"] for k in MVTEC_CATEGORIES]


def load_09_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """讀取 MVTEC 的 JSON 格式，轉換成 Detectron2 標準格式"""
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        image_file = os.path.join(image_dir, ann["file_name"])
        label_file = os.path.join(gt_dir, ann["file_name"].split('.')[0] + '.png') if gt_dir else None
        sem_label_file = os.path.join(semseg_dir, ann["file_name"].split('.')[0] + '.png') if semseg_dir else None
        segments_info = [_convert_category_id(x, meta) for x in ann.get("segments_info", [])]

        entry = {
            "file_name": image_file,
            "image_id": image_id,
            "segments_info": segments_info,
        }

        if label_file:
            entry["pan_seg_file_name"] = label_file
        if sem_label_file:
            entry["sem_seg_file_name"] = sem_label_file

        ret.append(entry)

    assert len(ret), f"No images found in {image_dir}!"
    return ret


def register_09_dataset(name, metadata, image_root, panoptic_json, instances_json=None, panoptic_root=None, semantic_root=None):
    """
    註冊 MVTEC 合併後的資料集：
    - `09_train`
    - `09_val`
    """
    DatasetCatalog.register(
        name,
        lambda: load_09_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="unified",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_MVTEC = {
    "09_train": (
        "09",  # `train` 只需要圖片
        None,  # `train` 沒有 ground truth
        None,  # `train` 不需要 semantic segmentation
        "09/09_train_INSTANCE.json",  # `train` 仍然需要 Instance JSON（即便 annotations 為空）
    ),
    "09_val": (
        "09",  # `test` 圖片
        "09",  # `test` ground truth
        "09/09_val_SEMANTIC.json",  # `test` semantic segmentation
        "09/09_val_INSTANCE.json",  # `test` instance segmentation
    ),
}


def get_metadata():
    """建立 metadata，定義類別顏色與 ID"""
    meta = {}
    thing_classes = [k["name"] for k in MVTEC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in MVTEC_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in MVTEC_CATEGORIES]
    stuff_colors = [k["color"] for k in MVTEC_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {cat["id"]: i for i, cat in enumerate(MVTEC_CATEGORIES) if cat["isthing"]}
    stuff_dataset_id_to_contiguous_id = {cat["id"]: i for i, cat in enumerate(MVTEC_CATEGORIES)}

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_09(root):
    """註冊所有 MVTEC 資料集"""
    metadata = get_metadata()
    for prefix, (image_root, panoptic_root, semantic_json, instance_json) in _PREDEFINED_SPLITS_MVTEC.items():
        register_09_dataset(
            prefix,
            metadata,
            os.path.join(root, image_root),
            panoptic_json=os.path.join(root, semantic_json) if semantic_json else None,
            instances_json=os.path.join(root, instance_json),
            panoptic_root=os.path.join(root, panoptic_root) if panoptic_root else None,
            semantic_root=os.path.join(root, panoptic_root) if panoptic_root else None,
        )


# 設定根目錄並註冊
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_09(_root)
