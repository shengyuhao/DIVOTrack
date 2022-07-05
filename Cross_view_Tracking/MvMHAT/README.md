# MvMHAT: Multi-view Multi-Human Association and Tracking

## Citation
The code is built with [MvMHAT](https://github.com/realgump/MvMHAT). Thanks for their great work.

## Get Started
The code was tested on Ubuntu 16.04, with Anaconda Python 3.6 and PyTorch v1.7.1. NVIDIA GPUs are needed for both training and testing. After install Anaconda:

0. Create a new conda environment：
~~~
   conda create -n MVMHAT python=3.6
~~~
And activate the environment:
~~~
   conda activate MVMHAT
~~~
1. Install pytorch:
~~~
   conda install pytorch=1.7.1 torchvision -c pytorch
~~~
2. Install the requirements:
~~~
   pip install -r requirements.txt
~~~
3. Download the pretrained model to promote convergence:
~~~
   cd $MVMHAT_ROOT/models
   wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O pretrained.pth
~~~

## Dataset Preparation
Use convert_divo.py to convert the DIVO dataset to the preferred model. In convert_divo.py, set "output_dir" to the path you want to store the dataset and run:
~~~
   python convert_divo.py
~~~ 
After the converting is finished, the dataset set should has the following format:
```
$output_dir
    └——————images
    |        └——————CircleRegion
    |        |  |——————CircleRegion_Drone_000001.jpg
    |        |  |——————CircleRegion_View1_000001.jpg
    |        |  |               ...
    |        └——————InnerShop
    |        |         ...
    |______train_gt
    |        └——————CircleRegion
    |        |  |——————Drone.txt
    |        |  |——————View1.txt
    |        |  |——————View2.txt
    |        └——————Innershop
    |        |  |——————Drone.txt
    |        |  |——————View1.txt
    |        |  |——————View2.txt
    |        |     ...
    |______test_gt
    |        └——————CircleRegion
    |        |  |——————Drone.txt
    |        |  |——————View1.txt
    |        |  |——————View2.txt
    |        |     ...
```

## Training
* First, set the ***ROOT_DIR*** and ***EX_ID*** in config.py to "$output_dir" and your own experiment ID respectively.
* train
```
   python train.py
```

## Inference
* Generate the feature npy file
```
   python inference.py --model "Your trained model name" --output_name "your save npy name (e.g. output.npy)"
```

The format of npy file is:
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


## Citation
    @inproceedings{gan2021mvmhat,
      title={Self-supervised Multi-view Multi-Human Association and Tracking},
      author={Yiyang Gan, Ruize Han, Liqiang Yin, Wei Feng, Song Wang},
      booktitle={ACM MM},
      year={2021}
    }
 

## References
- Portions of the code are borrowed from [Deep SORT](https://github.com/nwojke/deep_sort), thanks for their great work.
- Portions of the videos are borrowed from [Campus](http://web.cs.ucla.edu/~yuanluxu/research/mv_track.html) and [EPFL](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/), thanks for their contributions.

**More information is coming soon ...**
