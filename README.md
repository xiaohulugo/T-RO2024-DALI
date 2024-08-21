# T-RO2024-DALI
Official implementation of the T-RO paper "DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising"

## Introduction
Our code is based on ST3D (https://github.com/CVMI-Lab/ST3D/tree/master) which is based on OpenPCDet v0.3.0. More updates on OpenPCDet are supposed to be compatible with our code.

## Performance
### Waymo -> KITTI TASK
|     method        | AP_BEV@R40 | AP_3D@R40 | Model | 
|:-----------------:|:----------:|:---------:|:-----:|
|    DALI(CAD)      |    85.53   |    75.32  |[model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeMq80RN8K1Fsub3qWyfexMB5mIgb-eohHbs9iCMlTY9Pw?e=7ClPTt)|
|    DALI(Points)   |    85.46   |    75.10  |[model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeMq80RN8K1Fsub3qWyfexMB5mIgb-eohHbs9iCMlTY9Pw?e=7ClPTt)|


### nuScenes -> KITTI TASK
|     method        | AP_BEV@R40 | AP_3D@R40 | Model |
|:-----------------:|:----------:|:---------:|:-----:|
|    DALI(CAD)      |    83.43   |    69.09  | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeMq80RN8K1Fsub3qWyfexMB5mIgb-eohHbs9iCMlTY9Pw?e=7ClPTt)|
|    DALI(Points)   |    83.26   |    68.84  | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeMq80RN8K1Fsub3qWyfexMB5mIgb-eohHbs9iCMlTY9Pw?e=7ClPTt)|

## Installation

Please refer to the installation of ST3D [INSTALL.md](https://github.com/CVMI-Lab/ST3D/blob/master/docs/INSTALL.md) for the instructions.

## License

Our code is released under the MIT license.

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.3](https://github.com/open-mmlab/OpenPCDet/commit/e3bec15f1052b4827d942398f20f2db1cb681c01) and [St3D](https://github.com/CVMI-Lab/ST3D/tree/master). Thanks for their awesome codebase.

## Citation

If you find this project useful in your research, please consider to cite:
```
@article{lu2024dali,
  title={DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising},
  author={Lu, Xiaohu and Radha, Hayder},
  journal={IEEE Transactions on Robotics},
  year={2024},
  publisher={IEEE}
}
```
```
@inproceedings{yang2021st3d,
    title={ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection},
    author={Yang, Jihan and Shi, Shaoshuai and Wang, Zhe and Li, Hongsheng and Qi, Xiaojuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2021}
}
```
```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```
