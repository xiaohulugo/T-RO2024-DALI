# T-RO2024-DALI
Official implementation of the T-RO paper "DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising"

## Introduction
Our code is based on ST3D (https://github.com/CVMI-Lab/ST3D/tree/master) which is based on OpenPCDet v0.3.0. More updates on OpenPCDet are supposed to be compatible with our code.

## Model Zoo
### Waymo -> KITTI TASK
|                                                                                             |     method        | AP_BEV@R40 | AP_3D@R40 | 
|---------------------------------------------------------------------------------------------|:-----------------:|:----------:|:---------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(CAD&SN)    |    85.53   |    75.3   | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(CAD&ROS)   |    85.53   |    75.3   |
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(Points&ROS)|    85.53   |    75.3   | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(Points&ROS)|    85.53   |    75.3   | 

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you should achieve similar performance by training with the default configs. Note that the training Waymo data used in our work is version 1.0. 

### nuScenes -> KITTI TASK
|                                                                                             |     method        | AP_BEV@R40 | AP_3D@R40 | 
|---------------------------------------------------------------------------------------------|:-----------------:|:----------:|:---------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(CAD&SN)    |    85.53   |    75.3   | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(CAD&ROS)   |    85.53   |    75.3   |
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(Points&ROS)|    85.53   |    75.3   | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(Points&ROS)|    85.53   |    75.3   | 

### Waymo -> nuScenes TASK
|                                                                                             |     method        | AP_BEV@R40 | AP_3D@R40 | 
|---------------------------------------------------------------------------------------------|:-----------------:|:----------:|:---------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(CAD&SN)    |    85.53   |    75.3   | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(CAD&ROS)   |    85.53   |    75.3   |
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(Points&ROS)|    85.53   |    75.3   | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |   DALI(Points&ROS)|    85.53   |    75.3   | 

## Installation

Please refer to [INSTALL.md](https://github.com/CVMI-Lab/ST3D/blob/master/docs/INSTALL.md) for the installation.

## License

Our code is released under the MIT license.

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.3](https://github.com/open-mmlab/OpenPCDet/commit/e3bec15f1052b4827d942398f20f2db1cb681c01) and [St3D](https://github.com/CVMI-Lab/ST3D/tree/master). Thanks for their awesome codebase.

## Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{lu2024dali,
    title={DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising},
    author={Lu, Xiaohu and Radha, Hayder},
    booktitle={IEEE Transactions on Robotics},
    year={2024}
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
