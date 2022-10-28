## Acknowledgement:

This work contains 4 directories, all of they are modified based on code from Pytorch implementation of RandLA-Net: https://github.com/qiqihaer/RandLA-Net-pytorch

The original work released by the authors is implemented with tensorflow: https://github.com/QingyongHu/RandLA-Net

Thanks for the above works.

## Code Detail

Here are overall 4 directories containing works of different stages. If you want to run the code, I prepared a demo program in RandLA-Net directory.

RandLA-Net is for all 19 classes training and evaluation with LiDAR model and LiDAR+Camera model

RandLA-Net_closeset is for the first 15 classes training and evaluation with LiDAR+Camera model
and also evaluate with MSP(max softmax probablity)

RandLA-Net_openset treats the first 15 classes as known in training and evaluation with LiDAR+Camera model
and also evaluate with AUPR, AUROC.

RandLA-Net-incre is incremental model, from 15 known classes to all 19 classes training and evaluation with LiDAR model and LiDAR+Camera model.

To run demo, go with following steps:

1. Do as ./RandLA-Net/README.md required, setting the Python environment, Pytorch 1.8 is used in this work.
2. In ./RandLA-Net/output directory, there will be two models: checkpoint.tar for LiDAR uni-modal model(original RandLA-Net), checkpoint_lv.tar for transformer LiDAR-Camera model(This is too large for github-upload failed).
    To switch between multi-modal and uni-modal models, simply change require_img at Line 39 of ./RandLA-Net/helper_tool.py.
3. Demo data is in ./RandLA-Net/demo_data, change the path in ./RandLA-Net/semantickitti_testset.py, to run the demo, run ./RandLA-Net/main_demo.py.
4. After running demo, the prediction result will be saved, then you can run ./RandLA-Net/plot_3d.py with saved prediction result or ground truth for visualization.

(Not recommended)If you want to train the model and evaluate quantitatively, download the dataset from SemanticKITTI and KITTI official website and rebuild it with ./RandLA-Net/utils/data_prepare_semantickitti.py,
the dataset would take around 400G space. To train the model, run train.sh with multiple GPUs, only using one GPU is not enough to train the model. Since it is extremely computationally expensive, try to train it 
with the whole dataset is not recommended.

For more details check ./RandLA-Net/README.md