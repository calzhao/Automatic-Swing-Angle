# Automatic baseball bat angle detection in broadcast videos
>  Developed by Lingran Zhao and Ziyou Ren

<p align="center"> <img src='docs/detic.gif' align="center" height="300px"> </p>

Our code is built on the [Detic model](https://github.com/facebookresearch/Detic?tab=readme-ov-file). The implementation of bat orientation estimation can be divided into two steps:

1. Locate the keyframe using Dino features
2. Estimate the angle from the located keyframe using Detic

## Installation
See [installation instructions](INSTALL.md).

## Video dataset set up

An example video dataset would looks like this:

``````
Automatic-Swing-Angle/
└── videos-pirates
    └── demo_videos
        ├── 00F01C88-4AF3-4679-A7FC-0F41E060C061.mp4
        ├── 01F8CA5C-16A9-4959-A8D3-E849F7244077.mp4
        ├── 02853FB8-DCBA-4260-A237-ECA3C3C7C040.mp4
        ......
    └── amateur_demos
        ├── 1AF5E0A4-379C-4DD8-806A-1033903CF9C3.mp4
        ├── 2D1A7762-B1BB-493C-B86E-110096173EAE.mp4
        ├── 26EB9880-22F1-4B70-8BAB-10CB7FA63307.mp4
        ......
``````

## Evaluation

Then, we can run our code with the command `python evaluate.py --vid_source_root ./videos-pirates/amateur_demos`. By default, the prediction results will be saved to `BatEstimation_v2.csv`.

Note: The algorithm will mark the Angle_Conf as -1 and Angle_Confidence as 0 when there aren't any detection results on the predicted keyframe. (That said, we only provide results on videos where keyframes have a clear bat detection, ensuring higher precision on the positive samples)

## Demo
Run our demo using Colab (no GPU needed): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/183n8QE4UQEuu4MqqN7WYVU_Y22M4YEjL?usp=sharing)

The visualization of estimated bat angle :

<p align="center"> <img src='docs/perfect_bat.png' align="center" height="450px"> </p>