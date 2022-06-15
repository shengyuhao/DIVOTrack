cd src
python main.py tracking --exp_id divo --dataset divo --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1 --load_model ../models/crowdhuman.pth --num_epochs 30
cd ..