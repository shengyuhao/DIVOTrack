# Bag of Tricks and A Strong ReID Baseline

## Citation
The code is built with [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). Thanks for their great work.

The codes are expanded on a [ReID-baseline](https://github.com/L1aoXingyu/reid_baseline) , which is open sourced by our co-first author [Xingyu Liao](https://github.com/L1aoXingyu).

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd DIVOTrack/Cross_view_Tracking/StrongReID/` 

2. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [ignite=0.1.2](https://github.com/pytorch/ignite) (Note: V0.2.0 may result in an error)
    - [yacs](https://github.com/rbgirshick/yacs)

3. Prepare dataset
```
DIVOTrack
    └——————datasets
    |        └——————DIVO
    |           |——————ReID_format
    └——————${ROOT}
```

5. Prepare pretrained model
Put the model in ./models/. You can obtain the model from [Google Drive](https://www.google.com)


## Train

```bash
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('ReID_format')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

## Test

```
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('your train model path')"
```
The test will generate rsb_divo.npy to DIVOTrack/Cross_view_Tracking/StrongReID. If you want to change its name, modify in tools/test.py <br>
Format of rsb_divo.npy <br>
{ <br>
circleRegion:{Drone:[[fid,pid,lx,ly,w,h,1,0,0,0,feature],...],   View1:{...},   View2:{...}}, <br>
 innerShop:{Drone:{...}, View1:{...}, View2:{...}}, <br>
 ... <br>
 }

## Evaluation
Please refer to [Multi_view_Tracking](https://github.com/shengyuhao/DIVOTrack/tree/main/Multi_view_Tracking)

