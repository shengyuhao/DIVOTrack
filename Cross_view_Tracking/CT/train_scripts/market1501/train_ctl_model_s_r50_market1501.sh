python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/home/xqr/shengyuhao/ReID-Survey/toDataset/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/256_resnet50/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/home/xqr/shengyuhao/centroids-reid/models/resnet50-19c8e357.pth"