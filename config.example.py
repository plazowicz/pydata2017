import os

OUT_WEIGHTS_DIR = ''
WEIGHTS_DUMP_INTERVAL = 2000

CELEB_FACES_DIR = ''
GPU_ID = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
