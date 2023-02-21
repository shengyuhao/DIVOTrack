# CAMPUS
python train.py mot --exp_id CAMPUS --data_cfg '../src/lib/cfg/mvmhat_campus.json' --load_model "../models/fairmot_dla34.pth" / 
        --gpus 0,1,2,3 --batch_size 32 --num_epochs 30