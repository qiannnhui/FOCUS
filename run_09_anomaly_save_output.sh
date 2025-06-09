#!/bin/bash

BASE_DIR=./datasets/09
OUTPUT_DIR=./datasets/09/output
CONFIG_FILE=./configs/sod_dinov2_giant.yaml
MODEL_WEIGHTS=./weights/focus_giant_sod.pth

find "$BASE_DIR" -type f -path "*/test/09/*.jpg" | while read INPUT
do
    # 抓出像 D5_1@B2700800@SOD@DS00745F-Z502-T 這類型的目錄名稱
    TOP_DIR=$(echo "$INPUT" | cut -d'/' -f4)

    # 定義 output 路徑
    OUTPUT_PATH="$OUTPUT_DIR/$TOP_DIR"
    mkdir -p "$OUTPUT_PATH"

    echo "Processing $INPUT"
    echo "Saving results to $OUTPUT_PATH"

    CUDA_VISIBLE_DEVICES=2,3 python demo/demo.py \
        --config-file "$CONFIG_FILE" \
        --input "$INPUT" \
        --output "$OUTPUT_PATH" \
        --opts MODEL.WEIGHTS "$MODEL_WEIGHTS"
done
