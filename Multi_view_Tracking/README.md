# DIVOTrack: A Cross-View Dataset for Multi-Human Tracking in DIVerse Open Scenes
## Abstract
Cross-view multi-human tracking tries to link human subjects between frames and camera views that contain substantial overlaps. Although cross-view multi-human tracking has received increased attention in recent years, existing datasets still have several issues, including 1) missing real-world scenarios, 2) lacking diverse scenes, 3) owning a limited number of tracks, 4) comprising only static cameras, and 5) lacking standard benchmarks, which hinders the exploration and comparison of cross-view tracking methods.

To solve the above concerns, we present **DIVOTrack**: a new cross-view multi-human tracking dataset for **DIV**erse **O**pen scenes with dense tracking pedestrians in realistic and non-experimental environments. In addition, our DIVOTrack contains ten different types of scenarios and 550 cross-view tracks, which surpasses all existing cross-view human tracking datasets. Furthermore, our DIVOTrack contains videos that are collected by two mobile cameras and one unmanned aerial vehicle, allowing us to evaluate the efficacy of methods while dealing with dynamic views. Finally, we present a summary of current methodologies and a set of standard benchmarks with our DIVOTrack to provide a fair comparison and conduct a thorough analysis of current approaches.

- [Dataset Downloads](#Dataset Downlads)
  - [Dataset Structure](#Dataset Structure)
  - [Dataset Downloads](#Dataset Downloads)
- [Evaluation](#Evaluation)
- [Training Detector](#Training Detector)
- [Single-view Tracking](#Single-view Tracking)
- [Cross-view Tracking](#Cross-view Tracking)

The project is organized as:
```
.
├── ./Training_detector
├── ./Cross_view_tracking
├── ./Single_view_tracking
├── evaluation.py
└── Readme.md
```

## Dataset Description
### Dataset Structure
your path to *.npy <br>
Format of *.npy <br>
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


## Cross-view Tracking
```
python src/trackm.py --feature "your path to .npy file" --result_dir "your path to store the result"

```

## MOT Challenge Evaluation 
Please refer to [MOTChallengeEvalKit](https://github.com/shengyuhao/DIVOTrack/tree/main/MOTChallengeEvalKit_cv_test)
