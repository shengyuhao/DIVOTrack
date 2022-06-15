# Track to Detect and Segment: An Online Multi-Object Tracker (CVPR 2021)

## Installation

The code was tested on Ubuntu 18.04, with [Anaconda](https://www.anaconda.com/download). After installing Anaconda:

0. create a new conda environment. 

    ```
    conda create --name CenterTrack python=3.8
    conda activate CenterTrack
    conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch
    ```
    

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ```
    pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ```

3. Clone this repo:

    ```
    CenterTrack_ROOT=../Single_view_Tracking/CenterTrack
    git clone --recursive https://github.com/xingyizhou/CenterTrack $CenterTrack_ROOT
    ```
   
4. Install the requirements

    ```
    pip install -r requirements.txt
    ```
    
    
5. We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).
    ```
    git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
    cd DCNv2
    ./make.sh
    ```

6. Download pertained models for [monocular 3D tracking](https://drive.google.com/open?id=1e8zR1m1QMJne-Tjp-2iY_o81hn2CiQRt) and move them to `$CenterTrack_ROOT/models/`. 

## Data preparation
The data in the following structure:
```
DIVOTrack
    └——————datasets
    |        └——————DIVO
    |           |——————images
    |           |        └——————train
    |           |        └——————test
    |           └——————labels_with_ids
    |                    └——————train
    |                    └——————test  
    └——————${ROOT}
```

## Train
```
cd ${CenterTrack_ROOT}
cd src
python main.py tracking --exp_id trades_divo --dataset divo --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1 --save_point 10,20,25,30 --load_model ../models/crowdhuman.pth --clip_len 3 --max_frame_dist 10  --batch_size 12 --trades
```
or

```
sh ./experiments/divo_train.sh
```
## Inference
```
cd ${CenterTrack_ROOT}
cd src
python test.py tracking --exp_id trades_divo --dataset divo --pre_hm --ltrb_amodal --pre_thresh 0.5 --inference --clip_len 3 --track_thresh 0.4 --gpus 0 --trades --resume
```
or
```
sh ./experiments/divo_test.sh
```
## Evaluation

## License

CenterTrack is developed upon [CenterNet](https://github.com/xingyizhou/CenterNet). Both codebases are released under MIT License themselves. Some code of CenterNet are from third-parties with different licenses, please check the CenterNet repo for details. In addition, this repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics) for MOT evaluation and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for nuScenes evaluation and preprocessing. See [NOTICE](NOTICE) for detail. Please note the licenses of each dataset. Most of the datasets we used in this project are under non-commercial licenses.
