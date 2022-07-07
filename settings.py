import numpy as np

UINT_DTYPE = np.uint16
INT_DTYPE = np.uint16

# data storage will be at chunk size CHUNK_SIZE[i] // 2
# CHUNK_SIZE = "auto" # let dask choose chunk_size
CHUNK_SIZE = (500, 500, 500)
ANISOTROPY = (30, 6, 6)

# baseline related hyperparameters
CONNECTIVITY = 26
MAX_ERODE = 100
ERODE_DELTA = 100
NUM_ITER = 1

# point cloud inference hyperparameters
PC_DOWNSAMPLE_RADIUS = 200
PC_PRED_THRESHOLD = 0.5

# merging hyperparameters
NUM_DENDRITES = 50

# evaluation hyperparameters
THRESHOLD = (np.arange(1, 10) / 10).tolist()

# SLURM cluster config
SLURM_PROJECT_NAME = "dendrite"
SLURM_PARTITIONS = "partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100"
SLURM_CORES_PER_JOB = 48
SLURM_MEMORY_PER_JOB = "30GB"
SLURM_WALLTIME = "120:00:00"
SLURM_MIN_JOBS = 5
SLURM_MAX_JOBS = 50
SLURM_LOCAL_DIRECTORY = "/scratch/adhinart"
SLURM_SCALE_INTERVAL = "10s"  # how long until new scaling
SLURM_TARGET_DURATION = "60s"  # how long one task is supposed to take
SLURM_DASHBOARD_PORT = 8888
SLURM_WAIT_COUNT = 10
