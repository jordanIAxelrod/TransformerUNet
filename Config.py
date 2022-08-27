import torch
# Hyperparameters for training

patients = 4
err = 5e-2
img_size = 512
crop_px = 0.55
crop_per = 0.3

LEARNING_RATE = 1E-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = "data/images"
MSK_DIR = "data/masks"
NUM_WORKERS = 2

IMG_SIZE = 2048
N_PATCHES_DOWN = 64
N_PATCHES_UP = 256
DEPTH = 4
QKV_BIAS = True
N_LAYERS = 1
P = .1
ATTN_P = .1
CHAN_LIST = [3, 2, 4, 8, 16, 32]
MLP_RATIO = 4