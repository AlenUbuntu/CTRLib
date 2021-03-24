import os 
from yacs.config import CfgNode as CN 

_C = CN() 

# ------------------------------
# Misc
# ------------------------------
_C.SEED = 0
_C.DEVICE = 'cuda:0'
_C.WEIGHT = ''
_C.OUTPUT_DIR = '.'
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "dataset_catalog.py")
_C.MODEL_NAME = 'lr'

# ------------------------------
# DATASET CONFIGURATION
# ------------------------------
_C.DATASET = CN()
_C.DATASET.DIR = '/home/alan/Downloads/recommendation/Criteo'
_C.DATASET.CONFIG_DIR = '/home/alan/Documents/CTRLib/configs/criteo'
_C.DATASET.SCHEMA_CONF_FILE = 'schema.yaml'
_C.DATASET.FIELD_CONF_FILE = 'field.yaml'
_C.DATASET.NAME = 'Criteo'
_C.DATASET.TRAIN_VALID_TEST = (0.7, 0.1, 0.2)
_C.DATASET.CACHE_PATH = './criteo'
_C.DATASET.REBUILD_CACHE = False 
_C.DATASET.MAX_RECORDS_TO_READ = int(1e7)
_C.DATASET.LABEL_COUNTS = (970960, 29040)  # we compute label weights as 1./label_count for label 0 and label 1 respectively
_C.DATASET.IMB_STRATEGY = 'resample'  # can be 'resample', 'focal', '' (no strategy, valid for balanced dataset)

# ------------------------------
# DATALOADER
# ------------------------------
_C.DATALOADER = CN()
# Number of data loading threshold
_C.DATALOADER.NUM_WORKERS = 4
# Batch Size
_C.DATALOADER.BATCH_SIZE = 64  # split based on class ratio
# Pin Memory
_C.DATALOADER.PIN_MEMORY = True
# Number of Batches per epoch
_C.DATALOADER.NUM_BATCH = 1000

# ------------------------------
# OPTIMIZATION
# ------------------------------
_C.OPTIMIZER = CN()

_C.OPTIMIZER.NAME = 'adam'
_C.OPTIMIZER.LR = 1e-3  # learning rate for feature extractor

_C.OPTIMIZER.MAX_EPOCH = 20  # maximum epochs
_C.OPTIMIZER.MOMENTUM = 0.9

_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.WEIGHT_DECAY_BIAS = 0

_C.OPTIMIZER.GAMMA = 0.1 
_C.OPTIMIZER.BETAS = (0.9, 0.999)

_C.OPTIMIZER.WARMUP_FACTOR = 1.0 / 3
_C.OPTIMIZER.WARMUP_ITERS = 500
_C.OPTIMIZER.WARMUP_METHOD = 'linear'

_C.OPTIMIZER.LR_SCHEDULER = 'cosine_annealing_lr'
_C.OPTIMIZER.STEP_SIZE = 10

# ------------------------------
# Factorization Machine
# ------------------------------
_C.FM = CN()

_C.FM.LATENT_DIM = 512

# ------------------------------
# DNN YouTube
# ------------------------------
_C.DNN = CN()

_C.DNN.LATENT_DIM = 512
_C.DNN.AGGREGATE = 'sum'
_C.DNN.HIDDEN_DIMS_FRACTOR = (.5, .5, .5)
_C.DNN.DROPOUT_PROB = 0.5