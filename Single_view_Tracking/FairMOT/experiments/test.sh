cd src
python track.py mot --exp_name inference --load_model ../exp/mot/train/model_fairmot.pth --test_divo True --conf_thres 0.5 
cd ..