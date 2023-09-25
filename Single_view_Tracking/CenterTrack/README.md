# CenterTrack: Tracking Objects as Points

## Citation
The code is built with [CenterTrack](https://github.com/xingyizhou/CenterTrack). Thanks for their great work.

## Installation

Please refer to [INSTALL.md](https://github.com/shengyuhao/DIVOTrack/blob/main/Single_view_Tracking/CenterTrack/readme/INSTALL.md) for installation instructions.

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
    |           |        └——————train
    |           |        └——————test  
    |           └——————annoatations
    |           |        └——————train.json
    |           |        └——————test.json
    └——————${ROOT}
```
If your data does not has the "annotations" directory, please use the following command to generate
```
cd ${CenterTrack_ROOT}
cd src
python convert_divo_to_coco.py
```
 
## Train
The pre-trained model can download from [Google Drive](https://drive.google.com/file/d/1SD31FLwbXArcX3LXnRCqh6RF-q38nO7f) and put it into `./models/`
```
cd ${CenterTrack_ROOT}
cd src
python main.py tracking --exp_id divo --dataset divo --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1 --load_model ../models/crowdhuman.pth --num_epochs 30
```
or

```
sh ./experiments/divo_train.sh
```
## Inference
Make sure the "exp_id" is the same as the training one, the model will be directly loaded from the corresponding dir "exp/tracking/"exp_id"/model_last.pth".
```
cd ${CenterTrack_ROOT}
cd src
python test.py tracking --exp_id divo --dataset divo --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume
```
or
```
sh ./experiments/divo_test.sh
```
The result will be saved to "exp/tracking/"exp_id"/result_divo
## Evaluation
0. Download our final [model](https://drive.google.com/file/d/1-RM7T76RTPh8Q0h8Cn9ZdyhU-gnnPIuT/view?usp=share_link) and put it into `./exp/tracking/"exp_id"/`
1. Change the directory name from "result_divo" to "centertrack"
2. Make sure "centertrack" has the middle directory "data". (i.e. centertrack/data/circleRegion_Drone.txt instead of centertrack/circleRegion_Drone.txt)
3. Copy your result_divo to DIVOTrack/TrackEval/data/trackers/mot_challenge/divo
4. Go to DIVOTrack/TrackEval
5. See the instrcution on [TrackEval](https://github.com/shengyuhao/DIVOTrack/tree/main/TrackEval)

## License

CenterTrack is developed upon [CenterNet](https://github.com/xingyizhou/CenterNet). Both codebases are released under MIT License themselves. Some code of CenterNet are from third-parties with different licenses, please check the CenterNet repo for details. In addition, this repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics) for MOT evaluation and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for nuScenes evaluation and preprocessing. See [NOTICE](NOTICE) for detail. Please note the licenses of each dataset. Most of the datasets we used in this project are under non-commercial licenses.

