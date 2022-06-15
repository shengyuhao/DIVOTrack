cd src
python test_det.py mot --exp_id test_det --load_model '../exp/mot/train_det/model_det_last.pth' --data_cfg '../src/lib/cfg/data_divo.json' --gpus 0
cd ..