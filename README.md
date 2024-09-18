# T-RO2024-DALI
Official implementation of the T-RO paper "DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising" https://ieeexplore.ieee.org/document/10614889

## Introduction
Our code is based on ST3D (https://github.com/CVMI-Lab/ST3D/tree/master) which is based on OpenPCDet v0.3.0. More updates on OpenPCDet are supposed to be compatible with our code.

## Performance
### Waymo -> KITTI TASK
|     method        | AP_BEV@R40 | AP_3D@R40 | Model | 
|:-----------------:|:----------:|:---------:|:-----:|
|    DALI(CAD)      |    85.53   |    75.32  |[model](https://drive.google.com/file/d/1WzVQoue7JUiVmdXOWHJ_tLHGSqRLgV-a/view?usp=sharing)|
|    DALI(Points)   |    85.46   |    75.10  |[model](https://drive.google.com/file/d/1deZbQlRBEoYXaqkfPRqu60AEoZ7BbVfR/view?usp=sharing)|


### nuScenes -> KITTI TASK
|     method        | AP_BEV@R40 | AP_3D@R40 | Model |
|:-----------------:|:----------:|:---------:|:-----:|
|    DALI(CAD)      |    83.43   |    69.09  | [model](https://drive.google.com/file/d/1FfI7LhDIElICmCxAR6dqy0PBLXt24DkL/view?usp=sharing)|
|    DALI(Points)   |    83.26   |    68.84  | [model](https://drive.google.com/file/d/1Ni1s7ctSYvoJgIHWYEltfxarBqbwtjhF/view?usp=sharing)|

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
