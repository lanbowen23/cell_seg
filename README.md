# cell image segmentation  

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
and implemented with Keras functional API.
---

## Overview

### Data

The dataset is in folder `FISH`.

### Model

![img/unet-structure.jpg](img/unet-structure.jpg)

### Training

Train for 200 epoch. Early stop with patience of 10.

Loss function for the training is weighted crossentropy.

---

## Usage

### preprocess.ipynb
normalize raw image dataset, and transform raw instance label to 3 channel label (background, cell, boundary).

### full-experiment.ipynb
setting up train/val split directory and data, do training and prediction.

### evaluation.ipynb
evaluate prediction results using IoU metrics

### qualitative-analysis.ipynb
showing comparison of prediction images

---


## Acknowlegement

[unet4nuclei](https://github.com/carpenterlab/unet4nuclei)
