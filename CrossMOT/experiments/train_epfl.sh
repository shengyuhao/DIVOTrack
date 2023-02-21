# DIVOTrack
python train.py mot --exp_id DIVOTrack --data_cfg '../src/lib/cfg/divo.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30
