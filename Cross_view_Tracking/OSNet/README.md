# Torchreid
Torchreid is a library for deep-learning person re-identification, written in `PyTorch <https://pytorch.org/>` and developed for the ICCV'19 project, `Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/abs/1905.00953>`
### Quick Start

#### 1. Prepare dataset 
The structure of our dataset is as follow:
```
DIVOTrack
    └——————datasets
    |        └——————DIVO
    |           |——————images
    |           |        └——————train
    |           |        └——————test
    |           └——————labels_with_ids
    |           |        └——————train
    |           |        └——————test
    |           └——————ReID_format
    |                    └——————boundint_box_train
    |                    └——————boundint_box_test  
    └——————${ROOT}
```

The `bounding_box_test` is the cropped images set based on the [CenterNet](Traing_Detector/) detections.

#### 2. Install dependencies

```
conda create -n agw python=3.8
conda activate agw
pip install -r requirements.txt
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
python setup.py develop
```
  
#### 3. Train
Download the [pre-trained model](https) and put it into `./models/` 
To train a AGW model with on `./bounding_box_train/` with GPU device 0, run similarly:
```
sh Train_osnet.sh
```

#### 4. Test
Download our [model](https) and put it into `./log/ours/Experiment-AGW-baseline/`
To test a AGW model with on `./bounding_box_test/` with our model, run similarly:
```
sh Test_osnet.sh
```
The output is a `.npy` file contains `frame, pseudo_id, xmin, ymin, xmax, ymax, feature`.
This file is saved in `DIVOTrack/datasets/DIVO/npy/cross_view/OSNet/`
### Citation

This work is from:
```
    @inproceedings{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      booktitle={ICCV},
      year={2019}
    }
```

