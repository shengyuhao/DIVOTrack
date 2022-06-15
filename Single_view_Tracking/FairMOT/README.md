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
* You can download the pretrained model from [Google Drive](https://drive.google.com/file/d/1KIJMI6dUdXQrFqRxRZyfUDyi_8b2BnUO/view?usp=sharing).
After downloading, you should put the pretrained models in the following structure:
```
${ROOT}
   └——————models
           └——————ctdet_coco_dla_2x.pth
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

## Final Model

You can download our final model here: [FairMOT model](https://drive.google.com/file/d/1_Pf8Yet-VS6peDXBGddO73npcbSaEh3E/view?usp=sharing)

After downloading, you should put the final detection model in the following structure:
```
${ROOT}
   └——————exp
           └——————mot
                   └——————train_det
                            └——————model_det_last.pth

```


