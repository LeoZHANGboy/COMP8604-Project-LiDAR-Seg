# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236) on Semantic KITTI.

## Preparation

1. Clone this repository

2. Install some Python dependencies, such as scikit-learn. All packages can be installed with pip.

3. Install python functions. the functions and the codes are copied from the [official implementation with Tensorflow](https://github.com/QingyongHu/RandLA-Net).

```
sh compile_op.sh
```
4. Download the [Semantic KITTI dataset](http://semantic-kitti.org/dataset.html#download), and preprocess the data:
  ```
  python utils/data_prepare_semantickitti.py
  ```
   Note: Please change the dataset path in the 'data_prepare_semantickitti.py' with your own path.


## Train a model

  ```
  python main_SemanticKITTI.py
  ```

This repository implements the official version as much as possible. There are following differences:

1) We use Conv2D instead of ConvTranspose2D in the decoder part. Since the strides are [1,1] and conv kernels are [1,1], we think the Conv2D and ConvTranspose2D are the same in this condition.

2) We delate the bias in the Conv2D which is followed by a BN layer, since the bias is meaningless in the Conv2D if the Conv2D is followed by a BN layer.

3) We evaluate the network for one epoch after every 10 training epoches.
