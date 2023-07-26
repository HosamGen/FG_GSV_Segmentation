# Forty Guard Google Street View Segmentation
Code for the segmentation of google street view images for FortyGuard

This is a PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset (http://sceneparsing.csail.mit.edu/).

ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. Follow the link below to find the repository for our dataset and implementations on Caffe and Torch7:
https://github.com/CSAILVision/sceneparsing

Color encoding of semantic categories can be found here:
https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?usp=sharing


## Supported models
We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling. We have provided some pre-configured models in the ```config``` folder.

Encoder:
- MobileNetV2dilated
- ResNet18/ResNet18dilated
- ResNet50/ResNet50dilated
- ResNet101/ResNet101dilated
- HRNetV2 (W48)

Decoder:
- C1 (one convolution module)
- C1_deepsup (C1 + deep supervision trick)
- PPM (Pyramid Pooling Module, see [PSPNet](https://hszhao.github.io/projects/pspnet) paper for details.)
- PPM_deepsup (PPM + deep supervision trick)
- UPerNet (Pyramid Pooling + FPN head, see [UperNet](https://arxiv.org/abs/1807.10221) for details.)

## Environment
The code is developed under the following configurations.
- Hardware: >=4 GPUs for training, >=1 GPU for testing (set ```[--gpus GPUS]``` accordingly)
- Software: Ubuntu 16.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0***
- Dependencies: numpy, scipy, opencv, yacs, tqdm

## Quick start: Test on an image using our trained model 

1. First, install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the model checkpoints, including the encoder and decoder, these can be found in the ```download_checkpoint.sh``` file
   ```
   sh download_checkpoint.sh
   ```

3. This repo includes the evaluation section only, refer to the ```DemoSegmenter.ipynb" notebook for all the steps and changes from the original repo.


## Note: if needed, the original repo link is here: https://github.com/CSAILVision/semantic-segmentation-pytorch , including training, testing and evaluation.
