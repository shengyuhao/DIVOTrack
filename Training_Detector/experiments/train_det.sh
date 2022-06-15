cd src
python train.py mot --exp_id train_det --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/data_divo.json' --id_weight -1.0 --gpus 0 --batch_size 24 --lr 1.25e-4 --lr_step 90,120 --num_epochs 140
cd ..