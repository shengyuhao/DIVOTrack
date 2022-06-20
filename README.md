# DIVOTrack: A Cross-View Dataset for Multi-Human Tracking in DIVerse Open Scenes
## Abstract
Cross-view multi-human tracking tries to link human subjects between frames and camera views that contain substantial overlaps. Although cross-view multi-human tracking has received increased attention in recent years, existing datasets still have several issues, including 1) missing real-world scenarios, 2) lacking diverse scenes, 3) owning a limited number of tracks, 4) comprising only static cameras, and 5) lacking standard benchmarks, which hinders the exploration and comparison of cross-view tracking methods.

To solve the above concerns, we present **DIVOTrack**: a new cross-view multi-human tracking dataset for **DIV**erse **O**pen scenes with dense tracking pedestrians in realistic and non-experimental environments. In addition, our DIVOTrack contains ten different types of scenarios and 550 cross-view tracks, which surpasses all existing cross-view human tracking datasets. Furthermore, our DIVOTrack contains videos that are collected by two mobile cameras and one unmanned aerial vehicle, allowing us to evaluate the efficacy of methods while dealing with dynamic views. Finally, we present a summary of current methodologies and a set of standard benchmarks with our DIVOTrack to provide a fair comparison and conduct a thorough analysis of current approaches.

- Dataset Description
  - Dataset Structure
  - Dataset Downloads
- Training Detector
- Single-view Tracking
- Cross-view Tracking
- TrackEval
- Multi_view_Tracking
- MOTChallengeEvalKit_cv_test


The test result of the cross-view MOT baselone method *MvMHAT* on the DIVOTrack. 
![test.gif](asset/test.gif)
The ground truth of the DIVOTrak.
![gt.gif](asset/gt.gif)
## Dataset Description
We collect data in 10 different real-world scenarios, named: `'circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate'`. All
the sequences are captured by using 3 moving cameras: `'Drone', 'View1' 'View2'` and are manually synchronized. 
### Dataset Structure
The structure of our dataset as:
```
DIVOTrack
    └——————datasets
             └——————DIVO
                |——————images
                |        └——————annotations
                |        └——————dets    
                |        └——————train
                |        └——————test
                └——————labels_with_ids
                |        └——————train
                |        └——————test  
                |——————npy
                |——————ReID_format
                |        └——————bounding_box_test
                |        └——————bounding_box_train
                |        └——————query        
                └——————boxes.json  
```
### Dataset Downloads
The whole dataset can download from [GoogleDrive](https://drive.google.com/drive/folders/1QycDVFQticDUg0PE4uofUqULx_E_1GlF?usp=sharing). You can decompress each `.tar.gz` file in its folder.

## Training Detector
The training process of our detector is in ```./Training_detector/``` and the details can see from  [Training_detector/Readme.md]().
## Single-view Tracking
The implement of single-view tracking baseline methods is in ```./Single_view_Tracking``` and the details can see from [Single_view_Tracking/Readme.md]().
## Cross-view Tracking
The implement of cross-view tracking baseline methods is in ```./Cross_view_Tracking``` and the details can see from [Cross_view_Tracking/Readme.md]().
## TrackEval
We evaluation each single-view tracking baseline by ```./TrackEval```, and the details can see from [TrackEval/Readme.md](https://github.com/shengyuhao/DIVOTrack/tree/main/TrackEval#readme).
## Multi-view Tracking
The multi-view tracking results can get from `./Multi_view_Tracking`, the details can see from [Multi_view_Tracking/Readme.md](https://github.com/shengyuhao/DIVOTrack/tree/main/Multi_view_Tracking#readme)
## MOTChallengeEvalKit_cv_test
The cross-view evaluation can get from `./MOTChallengeEvalKit_cv_test`, and the details can see from [./MOTChallengeEvalKit_cv_test/Readme.md](https://github.com/shengyuhao/DIVOTrack/tree/main/MOTChallengeEvalKit_cv_test#readme)
