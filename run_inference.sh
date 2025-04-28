# for INPUT in ./datasets/VISA/images/*
for INPUT in ./datasets/DUTS/DUTS-TE/DUTS-TE-Image/*
do
    CUDA_VISIBLE_DEVICES=2,3 python demo/demo.py --config-file ./configs/sod_dinov2_giant.yaml   --input $INPUT   --output ./datasets/DUTS/output/   --opts MODEL.WEIGHTS ./weights/focus_giant_sod.pth
done
    # python demo/demo.py --config-file ./configs/sod_dinov2_giant.yaml   --input ./datasets/VISA/images/0006.JPG   --output ./datasets/VISA/output/   --opts MODEL.WEIGHTS ./weights/focus_giant_sod.pth