## Overview
This is a PyTorch implementation of BCsiNet inference. The key results in paper [Binary Neural Network Aided CSI Feedback in Massive MIMO System](http://arxiv.org/abs/2011.02692) can be reproduced.

## Requirements

The following requirements need to be installed.
- Python >= 3.7
- [PyTorch >= 1.6](https://pytorch.org/get-started/locally/)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model and setting can be found in our paper. On the other hand, Chao-Kai Wen provides a pre-processed COST2100 dataset, which we adopt in BCsiNet training and inference. You can download it from [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

#### B. Checkpoints Downloading
The checkpoints of our proposed BCsiNet can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1Hu_SUGiSn_h6K7Zf3mJEKw) (passwd: cism) or [Google Drive](https://drive.google.com/drive/folders/11fJ4MoG5fqoLHoEeoOUOZsPd4b8vg6fS?usp=sharing)

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── BCsiNet  # The cloned BCsiNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # COST2100 dataset downloaded following section A
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder downloaded following section B
│   │     ├── a2
│   │     ├── b3
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Key Results Reproduction

The key results reported in Table IV of the paper are presented as follows.

Compression Ratio | Methods | Scenario | NMSE | Params | Checkpoints Path
:--: | :--: | :-- | :-- | :-- | :--
1/4 | BCsiNet-A2 | indoor  | -17.25dB | 33K | a2/in01/model.pth
1/4 | BCsiNet-A2 | outdoor | -8.35dB  | 33K | a2/out01/model.pth
1/4 | BCsiNet-B3 | indoor  | -20.31dB | 33K | b3/in01/model.pth
1/4 | BCsiNet-B3 | outdoor | -9.77dB  | 33K | b3/out01/model.pth
1/8 | BCsiNet-A2 | indoor  | -12.38dB | 17K | a2/in02/model.pth
1/8 | BCsiNet-A2 | outdoor | -6.26dB  | 17K | a2/out02/model.pth
1/8 | BCsiNet-B3 | indoor  | -12.77dB | 17K | b3/in02/model.pth
1/8 | BCsiNet-B3 | outdoor | -6.86dB  | 17K | b3/out02/model.pth
1/16 | BCsiNet-A2 | indoor  | -8.99dB  | 8K | a2/in03/model.pth
1/16 | BCsiNet-A2 | outdoor | -4.17dB  | 8K | a2/out03/model.pth
1/16 | BCsiNet-B3 | indoor  | -10.71dB | 8K | b3/in03/model.pth
1/16 | BCsiNet-B3 | outdoor | -4.52dB  | 8K | b3/out03/model.pth
1/32 | BCsiNet-A2 | indoor  | -6.79dB  | 4K | a2/in04/model.pth
1/32 | BCsiNet-A2 | outdoor | -2.69dB  | 4K | a2/out04/model.pth
1/32 | BCsiNet-B3 | indoor  | -7.93dB  | 4K | b3/in04/model.pth
1/32 | BCsiNet-B3 | outdoor | -2.74dB  | 4K | b3/out04/model.pth

In order to reproduce the aforementioned key results, you need to download the given dataset and checkpoints. Moreover, you should arrange your project tree as instructed. An example of `Experiments/run.sh` can be found as follows.

``` bash
python /home/BCsiNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/a2/in01/model.pth' \
  --batch-size 200 \
  --workers 0 \
  --reduction 4 \
  --encoder-head A \
  --num-refinenet 2 \
  --cpu \
  2>&1 | tee log.out
```

> Note that the checkpoint must match exactly with the reduction, encoder_head and num_refinenet.

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Please refer to it for more information.

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet).
