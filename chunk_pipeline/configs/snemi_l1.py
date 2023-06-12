import glob
import numpy as np
import h5py

TASK = "snemi_l1"

GENERAL__ANISOTROPY = (30, 6, 6)

H5 = {"main": ("/mmfs1/data/bccv/dataset/snemi/pred_unetr_z5.h5", "main")}

ids = np.unique(h5py.File(H5["main"][0])[H5["main"][1]][:])
ids = ids[ids != 0]
L1__IDS = ids.tolist()
# set to 0 to use all points
L1__NOISE_STD = 15.0
L1__NUM_SAMPLE = 50000
l1_path = "/data/adhinart/L1-Skeleton"
# L1__DOWNSCALE_FACTOR = 0.001
L1__JSON_PATH = "/data/adhinart/dendrite/chunk_pipeline/configs/snemi_skeleton_config.json"  # changed downsample num to 3000

SLURM__NUM_PROCESSES_PER_JOB = 20
SLURM__MEMORY_PER_TASK = 5
