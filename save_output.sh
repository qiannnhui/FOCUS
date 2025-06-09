# for INPUT in ./datasets/VISA/images/*
OUTPUT_DIR=./datasets/09_val/output
# for INPUT in ./datasets/DUTS/DUTS-TE/DUTS-TE-Image/*
for INPUT in ./datasets/09_val/images/*
do
    CUDA_VISIBLE_DEVICES=2,3 python demo/demo.py --config-file ./configs/sod_dinov2_giant.yaml   --input $INPUT   --output $OUTPUT_DIR   --opts MODEL.WEIGHTS ./weights/focus_giant_sod.pth
done
    # python demo/demo.py --config-file ./configs/sod_dinov2_giant.yaml   --input ./datasets/VISA/images/0006.JPG   --output ./datasets/VISA/output/   --opts MODEL.WEIGHTS ./weights/focus_giant_sod.pth