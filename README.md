# VCOP
This is the implementation of our paper "[Self-supervised Spatiotemporal Learning via Video Clip Order Prediction](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf)". 

### Environments
* Ubuntu 16.04
* Python 3.6.1
* Pytorch 0.4.1

### Prerequisits
1. Clone the repository to your local machine.

    ```
    $ git clone https://github.com/xudejing/VCOP.git
    ```

2. Install the python dependency packages.

    **Default (when PyPI still serves old wheels):**

    ```
    $ pip install -r requirements.txt
    ```

    **Windows + Python 3.6, CPU — PyPI often has no `torch==0.4.1`:** install PyTorch from the official archive, then the rest.

    ```
    $ python -m pip install -r requirements-no-torch.txt
    $ python -m pip install https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl
    $ python -m pip install torchvision==0.2.1 --no-deps
    ```

    Use `--no-deps` for torchvision so it does not upgrade `numpy` (this repo pins `numpy==1.15.4`).
    BOBSL support uses OpenCV for video IO, so make sure `opencv-python==4.5.5.64` is installed when using the legacy Python 3.6 / torch 0.4.1 environment.

    If the torchvision wheel fails on PyPI, try: `python -m pip install torchvision==0.2.1 --no-deps -f https://download.pytorch.org/whl/torch_stable.html`

    **Note:** Using `-f https://download.pytorch.org/whl/torch_stable.html` without pinning the CPU URL may pull a large **CUDA** build (~600MB+); for CPU-only prefer the `whl/cpu/...` link above.

### Citation
If you find this code useful, please cite the following paper:
```
@inproceedings{xu2019self,
  title={Self-supervised Spatiotemporal Learning via Video Clip Order Prediction},
  author={Xu, Dejing and Xiao, Jun and Zhao, Zhou and Shao, Jian and Xie, Di and Zhuang, Yueting},
  booktitle={Computer Vision and Pattern Recognition (CVPR)}
  year={2019}
}
```
