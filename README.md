

# FOCUS: Towards Universal Foreground Segmentation
This repo is clone from [FOCUS](https://geshang777.github.io/focus.github.io/). 

## Overview

<p align="center">
  <img src="assets/focus_framework.png" width="90%" height="90%">
</p>

We utilize FOCUS as our zero-shot model to find out the foreground objects.

## Getting Started

### Environment Setup
There are 2 ways to setup, one is to directly download an docker image and create a container on your own, the other is to setup the environment step by step.
#### Docker File
1. Access image from `tmp2/handover` called `focus_env.tar`
2. Import
```bash
docker import focus_env.tar focus_img
```
3. Create your own container from the image
```bash
docker run -it --runtime=nvidia --gpus all \
  --device=/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools \
  --device=/dev/nvidia-modeset \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia0 \
  --device=/dev/nvidia1 \
  --device=/dev/nvidia2 \
  --device=/dev/nvidia3 \
  --shm-size=128g \
  --mount type=bind,source={place you want to mount},target=/home \
  --name focus_env \
  focus_img /bin/bash
```

#### Setup Steps
* We use CUDA 12.2 for implementation.
* Our code is built upon Pytorch 2.1.1, please make sure you are using PyTorch ≥ 2.1 and matched torchvision. Besides, please check PyTorch version matches that is required by Detectron2.
* This model should be trained with 48G memory, and 24G to evaluate, we use four 3080 GPUs to run.


```bash
#create environment
conda create --name focus python=3.8
conda activate focus
pip install -r requirements.txt

#install detectron2
git clone git@github.com:facebookresearch/detectron2.git # under your working directory
cd detectron2 && pip install -e . && cd ..

#install other dependencies
pip install git+https://github.com/cocodataset/panopticapi.git
cd third_party/CLIP
python -m pip install -Ue .
cd ../../

#compile CUDA kernel for MSDeformAttn
cd focus/modeling/pixel_decoder/ops && sh make.sh && cd ../../../../
```

### Quick Start


FOCUS provide an inference demo here if you want to try out the our model. You should download the weights from our [Model Zoo](https://drive.google.com/drive/folders/1IcyZnqc4vcsvSUcKb2llYGPt3ClFGjPl) first and run the following command. Make sure that you use the config file conrresbonding to the download weights.

The SOD model is downloaded to the server and is located at `/tmp2/qiannnhui/FOCUS/weights`.

```bash
python demo/demo.py --config-file path/to/your/config \
  --input path/to/your/image \
  --output path/to/your/output_file \
  --opts MODEL.WEIGHTS path/to/your/weights
```

### Prepare Datasets

You should download required dataset ([DUTS](http://saliencydetection.net/duts/), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [MVTEC](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff#data-download), [09](https://drive.google.com/file/d/1LFWsO6zPaOXyhbpNcInWKX6rB8xEcpCZ/view?usp=sharing)), into the `datasets` folder following.

These datasets are already in server at `/tmp2/qiannnhui/FOCUS/datasets/`.

```
datasets/
├── DUTS
│   ├── DUTS-TE
│   └── DUTS-TR
├── ECSSD
│   ├── ground_truth_mask
│   └── images
├── 09
│   ├── ground_truth_mask
│   └── images
├── mvtec_good
│   ├── ground_truth_mask
│   └── images
├── visa_good
│   ├── ground_truth_mask
│   └── images
│
```

and run the corresponding dataset preparation script by running:

```bash
python utils/prepare/prepare_<dataset>.py

# e.g. python utils/prepare/prepare_duts.py
# e.g. python utils/prepare/prepare_ecssd.py
# e.g. python utils/prepare/prepare_09_val.py
# e.g. python utils/prepare/prepare_mvtec_good.py
# e.g. python utils/prepare/prepare_visa_good.py
```

There are some datasets in the anomaly detection form like MVTEC, VISA. If you want to turn it as the form above, try the script in `datasets/preprocess_09/anomaly_to_FGBG`. We only provided the dataset with only one segmentation groundtruth mask for each class, named end with "good".

Furthermore, the "09" dataset didn't map the size of input image and ground image well, and we'll have to turn its multiple class mask to a single class mask. Thus, we provided a script to preprocess it, including prepare it's json file.
```bash
./preprocess_09.sh
```


### Prepare Pretrained Weights

These pretrained weight has already been downloaded and processed on the server under the directory `/tmp2/FOCUS/ckpt/`. 
Or you can download it using the command below:

download pre-trained DINOv2 weights by:

```bash
#dinov2-g
wget -P ./ckpt https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth
```

and run the following line to convert DINOv2 weights into detectron2 format while prepare ResNet weights for edge enhancer

```bash
#dinov2-g
python utils/convert_dinov2.py ./ckpt/dinov2_vitg14_reg4_pretrain.pth ./ckpt/dinov2_vitg14_pretrain_updated.pkl
```


All of the input below is the configuration file in `./configs/sod_dinov2_giant.yaml`.

## Training
```bash
python train_net.py \
--config-file path/to/your/config \
--num-gpus NUM_GPUS
```

We didn't run this command from [Focus](https://geshang777.github.io/focus.github.io/) yet due to the zero-shot training.

## Evaluation

```bash
python train_net.py --eval-only \
--config-file path/to/your/config \
--num-gpus NUM_GPUS \
MODEL.WEIGHTS path/to/your/weights
```

Note that there is a script to run this command called `run_inference.sh`. If there's an assertion fault in `detectron2`, simply comment it out.

The desired output is the evaluation results in the command line.

## Draw the Output
```bash
./save_output.sh
```

The input and output will be the input images and the prediction, the directory should be set in the script by yourself.

## Acknowledgements
[FOCUS](https://geshang777.github.io/focus.github.io/) is built upon [Mask2Former](https://github.com/facebookresearch/Mask2Former), [CLIP](https://github.com/openai/CLIP), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [OVSeg](https://github.com/facebookresearch/ov-seg/), and [detectron2](https://github.com/facebookresearch/detectron2). We express our gratitude to the authors for their remarkable work.
