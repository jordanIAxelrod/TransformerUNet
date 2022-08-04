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