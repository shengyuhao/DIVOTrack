# FairMOT
## Requirements

* Clone this repo, and we'll call the `DIVOTrack_github/Single_view_Tracking/FairMOT/` as ${ROOT}
* Install dependencies. We use python 3.8 and pytorch >= 1.7.0
```
conda create -n fairmot
conda activate fairmot
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch
cd ./Training_Detector
pip install cython
pip install -r requirements.txt
```
* We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).
```
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```

## Data preparation
The data in the following structure:
```
DIVOTrack_github
    └——————datasets
    |        └——————DIVO
    |           |——————images
    |           |        └——————train
    |           |        └——————test
    |           └——————labels_with_ids
    |                    └——————train
    └——————${ROOT}
```

## Pretrained model
* You can download the pretrained model from [Google Drive](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view).
After downloading, you should put the pretrained models in the following structure:
```
${ROOT}
   └——————models
           └——————fairmot_dla34.pth
```

## Training

* Download the training data
* To train the model in the paper, run this command:

```train
sh experiments/train.sh
```

## Inference
* To get the inference results, run:

```test
sh experiments/test.sh
```
* The result with be save to ../result/exp_id/

## Evaluation
1. Change the directory name from "result_divo" to "fairmot"
2. Make sure "centertrack" has the middle directory "data". (i.e. fairmot/data/circleRegion_Drone.txt instead of fairmot/circleRegion_Drone.txt)
3. Resize the result files by "resize.py"
4. Copy your result_divo to DIVOTrack/TrackEval/data/trackers/mot_challenge/divo
5. Go to DIVOTrack/TrackEval
6. See the instruction on [TrackEval](https://github.com/shengyuhao/DIVOTrack/tree/main/TrackEval)

## Final Model

You can download our final model here: [FairMOT model](https://drive.google.com/file/d/1TA7uHJtweHc0BYdH3eo3vFdtMahHutxi/view?usp=share_link)

After downloading, you should put the final detection model in the following structure:
```
${ROOT}
   └——————exp
           └——————mot
                   └——————train_det
                            └——————model_det_last.pth

```


