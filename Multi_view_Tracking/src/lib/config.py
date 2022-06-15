# dataset
from torch import detach


TRAIN_DATASET = ['circleRegion', 'innerShop', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
TEST_DATASET = ['circleRegion', 'innerShop', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
# park only
# TRAIN_DATASET = ['park']
# TEST_DATASET = ['park']
FRAMES = 2
VIEWS = 3

# training
TRAIN_GPUS = '1'
EX_ID = '0313_all_dataset'
LOSS = ['pairwise', 'triplewise', 'reid']
W_MVMHAT = 1
W_REID = 2
LEARNING_RATE = 1e-5
MAX_EPOCH = 8
RE_ID = 1
RE_ID_DIMENSION = 1024 if RE_ID == 1 else 0

NETWORK = 'resnet'
# if RE_ID:
#     TRAIN_RESUME = "./models/" + RE_ID + '.pth'
MODEL_SAVE_NAME = "./output/" + EX_ID + "/"

# parameters
MARGIN = 0.8
DATASET_SHUFFLE = 0
LOADER_SHUFFLE = 1

# inference
INF_ID = 'model_1'
DISPLAY = 0
INFTY_COST = 1e+5
RENEW_TIME = 1

# detach
detach_choice = 1