# MTSR
The pytorch code of hyperspectral image super-resolution method MTSR.

The code will be released after acceptance.

## Requirements
* Python 3.10
* Pytorch 2.1.1
* CUDA 11.8

## Preparation
To get the training set, validation set and testing set, refer to SSPSR to download the mcodes for cropping the hyperspectral image.

## Training
To train MTSR, run the following command.<br>
```
sh demo.sh
```
## Testing
run the the following command.<br>
```
sh test_demo.sh
```
## References
* MSDformer：[MSDformer: Multi-Scale Dilated Residual Network for Hyperspectral Image Super-Resolution](https://github.com/Tomchenshi/MSDformer.git)
* SSPSR：[Learning Spatial-Spectral Prior for Super-Resolution of Hyperspectral Imagery](https://github.com/junjun-jiang/SSPSR.git)]
