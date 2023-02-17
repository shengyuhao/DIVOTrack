# EPFL
python train.py mot --exp_id EPFL --data_cfg '../src/lib/cfg/EPFL.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30 --zero_start  

# DIVOTrack
python train.py mot --exp_id DIVOTrack --data_cfg '../src/lib/cfg/divo.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30

# MvMHAT
python train.py mot --exp_id MvMHAT --data_cfg '../src/lib/cfg/mvmhat.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30

# CAMPUS
python train.py mot --exp_id CAMPUS --data_cfg '../src/lib/cfg/mvmhat_campus.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30

# Wildtrack
python train.py mot --exp_id CAMPUS --data_cfg '../src/lib/cfg/wildtrack.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30