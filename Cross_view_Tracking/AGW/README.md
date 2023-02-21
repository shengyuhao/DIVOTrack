# ReID-Survey with a Powerful AGW Baseline
Deep Learning for Person Re-identification:  A Survey and Outlook. PDF with supplementary materials. [TPAMI](https://ieeexplore.ieee.org/abstract/document/9336268)

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
Download the [pre-trained model](https://drive.google.com/file/d/1kPr3jWutqkW7AlqBOy0XqjfXaupslvti/view?usp=share_link) and put it into `./models/` 
To train a AGW model with on `./bounding_box_train/` with GPU device 0, run similarly:
```
sh Train-AGW.sh
```

#### 4. Test
Download our [model](https://drive.google.com/file/d/1kPr3jWutqkW7AlqBOy0XqjfXaupslvti/view?usp=share_link) and put it into `./log/ours/Experiment-AGW-baseline/`
To test a AGW model with on `./bounding_box_test/` with our model, run similarly:
```
sh Test-AGW.sh
```
The out put is a `.npy` file contains `frame, pseudo_id, xmin, ymin, xmax, ymax, feature`.
This file is saved in `DIVOTrack/datasets/DIVO/npy/cross_view/AGW/`
<br><br> The format of npy file is:
```
{
circleRegion:{
    Drone:[[fid,pid,lx,ly,w,h,1,0,0,0,feature],...],   
    View1:[...],   
    View2:[...]
}, 
 innerShop:{
    Drone:[[fid,pid,lx,ly,w,h,1,0,0,0,feature],...],   
    View1:[...],   
    View2:[...]
}, 
 ...
 }
```

## Evaluation
Please refer to [Multi_view_Tracking](https://github.com/shengyuhao/DIVOTrack/tree/main/Multi_view_Tracking)


### Citation

Please kindly cite this paper in your publications if it helps your research:
```
@article{pami21reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
}
```

Contact: mangye16@gmail.com
