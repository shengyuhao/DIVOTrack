# EPFL
cd src
python train.py mot --exp_id EPFL --data_cfg '../src/lib/cfg/EPFL.json' --load_model "../models/fairmot_dla34.pth" --gpus 0,1,2,3 --batch_size 32 --num_epochs 30 --zero_start
