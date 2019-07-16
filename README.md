# PRNet-Depth-Generation 

---

## Introduction

A implementaion of depth generation based on [PRNet](https://github.com/YadiraF/PRNet), which was used in the paper ***Exploiting Temporal and Depth Information for Multi-frame Face Anti-spoofing***

## Prerequisite

* Python 3.6 (numpy, skimage, scipy)

* TensorFlow >= 1.4

  Optional:

* dlib (for detecting face.  You do not have to install if you can provide bounding box information. Other face detectors are ok if you want.)

* opencv2 (for showing results)

* Download the PRN trained model at [BaiduDrive](https://pan.baidu.com/s/10vuV7m00OHLcsihaC-Adsw) or [GoogleDrive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing), and put it into `Data/net-data`

## Test

> python Generate_Depth_Image.py

## License

Code: under MIT license.

## Citation

If you use this code, please consider citing:

```
@inProceedings{wang2018fastd,
  title     = {Exploiting Temporal and Depth Information for Multi-frame Face Anti-spoofing},
  author    = {Zezheng Wang, Chenxu Zhao, Yunxiao Qin, Qiusheng Zhou, Guojun Qi, Jun Wan, Zhen Lei},
  booktitle = {arXiv:1811.05118},
  year      = {2018}
}
@inProceedings{feng2018prn,
  title     = {Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network},
  author    = {Yao Feng, Fan Wu, Xiaohu Shao, Yanfeng Wang, Xi Zhou},
  booktitle = {ECCV},
  year      = {2018}
}
```

## Acknowledgements
Thanks *Yao Feng etc.* for their [PRNet](https://github.com/YadiraF/PRNet).
