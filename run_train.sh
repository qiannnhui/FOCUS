CUDA_VISIBLE_DEVICES=0,1,2,3
python train_net.py --eval-only --config-file ./configs/sod_dinov2_giant.yaml --num-gpus 4