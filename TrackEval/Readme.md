
# TrackEval

## Citation
The code is built with [TrackEval](https://github.com/JonathonLuiten/TrackEval). Thanks for their great work.

## Requirements
 Code tested on Python 3.7.
 
 - Minimum requirements: numpy, scipy
 - For plotting: matplotlib
 - For segmentation datasets (KITTI MOTS, MOTS-Challenge, DAVIS, YouTube-VIS): pycocotools
 - For DAVIS dataset: Pillow
 - For J & F metric: opencv_python, scikit_image
 - For simples test-cases for metrics: pytest

use ```pip3 -r install requirements.txt``` to install all possible requirements.

use ```pip3 -r install minimum_requirments.txt``` to only install the minimum if you don't need the extra functionality as listed above.

## Data preparation
The data is in the following structure:
```
TrackEval/data
    └——————gt
    |        └——————mot_challenge
    |           |——————divo
    |           |        └——————circleRegion_Drone
    |           |        		└——————gt
    |           |        			└——————gt.txt
    |           |        		└——————seqinfo.ini
    |           |        └——————circleRegion_View1
    |           |        		└——————gt
    |           |        			└——————gt.txt
    |           |        		└——————seqinfo.ini
    |		|		...
    |           |——————seqmaps
    |           |        └——————divo.txt
    └——————trackers
    |        └——————mot_challenge
    |           |——————divo
    |           |        └——————"your_result"
    |           |        		└——————data
    |           |        			└——————circleRegion_Drone.txt
    |           |        			└——————circleRegion_View1.txt
    |           |        			└——————circleRegion_View2.txt
    |      	|					    ...
```
Every single line in gt.txt has the format:
```
frame_id,person_id,lx,ly,w,h,1,1,1
```
Remember the tracking result is in "your_result"/data instead of "your_result".
Every single line in "your_result"/data/.txt should has the format:
```
frame_id,person_id,lx,ly,w,h,-1,-1,-1,-1
```


## Evaluation				
```
python scripts/run_mot_challenge.py --GT_FOLDER './data/gt/mot_challenge' --TRACKERS_FOLDER './data/trackers/mot_challenge' --BENCHMARK divo --TRACKERS_TO_EVAL your_result
```
The output will be stored to "./data/trackers/mot_challenge/divo/your_result" and has the following four files: <br>
pedestrian_detailed.csv <br>
pedestrian_plot.png <br>
pedestrian_plot.pdf <br>
pedestrian_summary.txt <br>

## License

TrackEval is released under the [MIT License](LICENSE).

## Contact

If you encounter any problems with the code, please contact [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/) ([luiten@vision.rwth-aachen.de](mailto:luiten@vision.rwth-aachen.de)).
If anything is unclear, or hard to use, please leave a comment either via email or as an issue and I would love to help.

## Dedication

This codebase was built for you, in order to make your life easier! For anyone doing research on tracking or using trackers, please don't hesitate to reach out with any comments or suggestions on how things could be improved.

## Contributing

We welcome contributions of new metrics and new supported benchmarks. Also any other new features or code improvements. Send a PR, an email, or open an issue detailing what you'd like to add/change to begin a conversation.

## Citing TrackEval

If you use this code in your research, please use the following BibTeX entry:

```BibTeX
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```

Furthermore, if you use the HOTA metrics, please cite the following paper:

```
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

If you use any other metrics please also cite the relevant papers, and don't forget to cite each of the benchmarks you evaluate on.
