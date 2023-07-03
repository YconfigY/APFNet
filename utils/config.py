import os
import yaml
from yacs.config import CfgNode as CN

__C = CN()
cfg = __C
# Base config files
__C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
__C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
__C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
__C.DATA.DATA_PATH = '/home/ubuntu/data/~/'

# Dataset name
__C.DATA.TRAIN_DATASET = 'RGBT234'
__C.DATA.TEST_DATASET = 'GTOT'
__C.DATA.CHALLENGE = ['FM', 'SV', 'OCC', 'ILL', 'TC']
# Input image size
__C.DATA.IMG_SIZE = 224
__C.DATA.TEST_SIZE = 107
__C.DATA.TEST_BATCH_SIZE = 256
__C.DATA.PADDING = 16
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
__C.DATA.PIN_MEMORY = True
# Number of data loading threads
__C.DATA.NUM_WORKERS = 16

__C.DATA.BATCH_FRAMES = 8
__C.DATA.BATCH_POS = 32
__C.DATA.BATCH_NEG = 96
__C.DATA.OVERLAP_POS = [0.7, 1]
__C.DATA.OVERLAP_NEG = [0, 0.5]

# Training examples sampling
__C.DATA.TRANS_POS = 0.1
__C.DATA.TRANS_NEG = 2
__C.DATA.SCALE_POS = 1.3
__C.DATA.SCALE_NEG = 1.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
__C.MODEL = CN()
__C.MODEL.NAME = 'APFNet-RepVGG-5_26'
# Model type
__C.MODEL.PRETRAINED = './data/models/RepVGG-A2-train.pth' #'./data/models/imagenet-vgg-m.mat'
# Checkpoint to resume, could be overwritten by command line argument
__C.MODEL.RESUME = None # './resume/stage_1/GTOT/ILL/ILL.pth'
__C.MODEL.DEVICE = 'CUDA'
# Train or track layer
__C.MODEL.STAGE_TYPE = 1   ##### STAGE
__C.MODEL.STAGE = CN()
__C.MODEL.STAGE.LR_LAYER = ['parallel','fc']
__C.MODEL.STAGE.LR_MULT = [30, 10]  # ILL:[20, 5] other:[10, 5]

__C.MODEL.STAGE_1 = CN()
__C.MODEL.STAGE_1.LR_LAYER = ['parallel','fc']
__C.MODEL.STAGE_1.LR_MULT = [30, 10]

__C.MODEL.STAGE_2 = CN()
__C.MODEL.STAGE_2.LR_LAYER = ['ensemble','fc']
__C.MODEL.STAGE_2.LR_MULT = [10, 5]

__C.MODEL.STAGE_3 = CN()
__C.MODEL.STAGE_3.LR_LAYER = ['transformer', 'fc', 'layer', 'parallel', 'ensemble']
__C.MODEL.STAGE_3.LR_MULT = [10, 5, 1, 1, 1, 1]

__C.MODEL.STAGE_TEST = CN()
__C.MODEL.STAGE_TEST.LR_LAYER = ['fc4', 'fc5', 'fc6']
__C.MODEL.STAGE_TEST.LR_MULT = [5, 5, 10]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
__C.TRAIN = CN()
__C.TRAIN.START_EPOCH = 0
__C.TRAIN.EPOCHS = 200
__C.TRAIN.WARMUP_EPOCHS = 0
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Clip gradient norm
__C.TRAIN.CLIP_GRAD = 10.0
# Auto resume from latest checkpoint
__C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
__C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
__C.TRAIN.USE_CHECKPOINT = False

# Branch challenge
__C.TRAIN.CHALLENGE = 'FM'  # ['FM', 'SV', 'OCC', 'ILL', 'TC']
__C.TRAIN.BATCH_ACCUM = 50

__C.TRAIN.SNAPSHOT = './snapshot/stage'
# LR
__C.TRAIN.LR = CN()
__C.TRAIN.LR.BASE = 0.0001
__C.TRAIN.LR.WARMUP = 0.0
__C.TRAIN.LR.MIN = 0.0
__C.TRAIN.LR.DECAY = []
__C.TRAIN.LR.GAMMA = 0.1

# LR scheduler
__C.TRAIN.LR_SCHEDULER = CN()
__C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
__C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
__C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
__C.TRAIN.OPTIMIZER = CN()
__C.TRAIN.OPTIMIZER.NAME = 'sgd'
# Optimizer Epsilon
__C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
__C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
__C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Tracker settings
# -----------------------------------------------------------------------------
__C.TRACK = CN()
__C.TRACK.RESULTS_PATH = './results'
__C.TRACK.BATCH_SIZE = 256
__C.TRACK.BATCH_NEG_CAND = 1024

# Candidates sampling
__C.TRACK.CAND = CN()
__C.TRACK.CAND_SAMPLES_NUM = 256
__C.TRACK.CAND_TRANS = 0.6
__C.TRACK.CAND_TRANS_LIMIT = 1.5
__C.TRACK.CAND_SCALE = 1.05

# Training examples sampling
__C.TRACK.EXAMPLES = CN()
__C.TRACK.EXAMPLES.TRANS_POS = 0.1
__C.TRACK.EXAMPLES.TRANS_NEG = 2
__C.TRACK.EXAMPLES.TRANS_NEG_INIT = 1
__C.TRACK.EXAMPLES.SCALE_POS = 1.3
__C.TRACK.EXAMPLES.SCALE_NEG = 1.3
__C.TRACK.EXAMPLES.SCALE_NEG_INIT = 1.6

# Bounding box regression
__C.TRACK.BBOX_REG = CN()
__C.TRACK.BBOX_REG.SAMPLES_NUM = 1024 # default is 1000
__C.TRACK.BBOX_REG.OVERLAP = [0.6, 1]
__C.TRACK.BBOX_REG.TRANS = 0.3
__C.TRACK.BBOX_REG.SCALE = 1.6
__C.TRACK.BBOX_REG.ASPECT = 1.1

__C.TRACK.TRAIN = CN()
# Initial training
__C.TRACK.TRAIN.INIT = CN()
__C.TRACK.TRAIN.INIT.LR = 0.0005
__C.TRACK.TRAIN.INIT.MAX_ITER = 50 
__C.TRACK.TRAIN.INIT.NUN_POS = 512  # default is 500
__C.TRACK.TRAIN.INIT.NUM_NEG = 5120  # default is 5000
__C.TRACK.TRAIN.INIT.OVERLAP_POS = [0.7, 1]
__C.TRACK.TRAIN.INIT.OVERLAP_NEG = [0, 0.5]

# Online training
__C.TRACK.TRAIN.UPDATE = CN()
__C.TRACK.TRAIN.UPDATE.LR = 0.001
__C.TRACK.TRAIN.UPDATE.MAX_ITER = 15
__C.TRACK.TRAIN.UPDATE.NUN_POS = 64  # default is 50
__C.TRACK.TRAIN.UPDATE.NUM_NEG = 256  # default is 200
__C.TRACK.TRAIN.UPDATE.OVERLAP_POS = [0.7, 1]
__C.TRACK.TRAIN.UPDATE.OVERLAP_NEG = [0, 0.3]

# Update criteria
__C.TRACK.TRAIN.UPDATE.LONG_INTERVAL = 10
__C.TRACK.TRAIN.UPDATE.NUM_FRAMES_LONG = 100
__C.TRACK.TRAIN.UPDATE.NUM_FRAMES_SHORT = 30

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
__C.AUG = CN()
__C.AUG.FLIP = True
__C.AUG.ROTATE = 30
__C.AUG.BLUR = 7

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
__C.TEST = CN()
# Batch size

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
__C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
__C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
__C.TAG = 'default'
# Frequency to save checkpoint
__C.SAVE_FREQ = 20
# Frequency to logging info
__C.PRINT_FREQ = 10
# Fixed random seed
__C.SEED = 0
# Perform evaluation only, overwritten by command line argument
__C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
__C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
__C.LOCAL_RANK = 0


def update_config(config):
    config.defrost()
    # merge from specific arguments
    # if args.challenge:
    #     config.TRAIN.CHALLENGE = args.challenge
    # if args.stage:
    #     config.MODEL.STAGE_TYPE = args.stage
    # if args.epoch:
    #     config.TRAIN.EPOCHS = args.epoch
        
    if config.TRAIN.CHALLENGE == 'ILL':
        config.MODEL.STAGE.LR_MULT = [20, 5]    
    
    if config.MODEL.STAGE_TYPE == 1:
        config.MODEL.STAGE = config.MODEL.STAGE_1
    elif config.MODEL.STAGE_TYPE == 2:
        config.MODEL.STAGE = config.MODEL.STAGE_2
    elif config.MODEL.STAGE_TYPE == 3:
        config.MODEL.STAGE = config.MODEL.STAGE_3
    elif config.MODEL.STAGE_TYPE == 4:
        config.MODEL.STAGE = config.MODEL.STAGE_TEST
        
    # set local rank for distributed training
    # config.LOCAL_RANK = args.local_rank
    # output folder
    config.freeze()


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = __C.clone()
    update_config(config)

    return config