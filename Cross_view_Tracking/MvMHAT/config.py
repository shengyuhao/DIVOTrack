# dataset
TRAIN_DATASET = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
TEST_DATASET = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
FRAMES = 2
VIEWS = 3
ROOT_DIR = '/mnt/sdb/mvmhat'

# training
TRAIN_GPUS = '0,1,2,3'
EX_ID = 'lpy'
LOSS = ['pairwise', 'triplewise']
LEARNING_RATE = 1e-5
MAX_EPOCH = 10


NETWORK = 'resnet'
RE_ID = 0
if RE_ID:
    TRAIN_RESUME = "./models/" + RE_ID + '.pth'
MODEL_SAVE_NAME = "./models/" + EX_ID + '.pth'

# parameters
MARGIN = 0.5
DATASET_SHUFFLE = 0
LOADER_SHUFFLE = 1

# inference
INF_ID = 'model'
DISPLAY = 0
INFTY_COST = 1e+5
RENEW_TIME = 30
DETECTION_DIR = '../../datasets/DIVO/images/dets/detection_results/'
