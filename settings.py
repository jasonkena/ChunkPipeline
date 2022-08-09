import numpy as np
import math

UINT_DTYPE = np.uint16
INT_DTYPE = np.int16

# data storage will be at chunk size CHUNK_SIZE[i] // 2
# CHUNK_SIZE = "auto" # let dask choose chunk_size
CHUNK_SIZE = (500, 500, 500)
# for den_seg
ANISOTROPY = (30, 6, 6)
# for human and mouse
# ANISOTROPY = (30, 8, 8)

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

# skeletonization hyperparameters

KIMI_DOWNSAMPLE_RADIUS = 30
KIMI_PARAMS = {
    "teasar_params": {
        "scale": 1.5,
        "const": 300,
        "soma_detection_threshold": 1500,
        "soma_acceptance_threshold": 3500,
        "soma_invalidation_scale": 2,
        "soma_invalidation_const": 300,
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "max_paths": None,
    },
    "dust_threshold": 1000,
}

# evaluation hyperparameters
THRESHOLD = (np.arange(1, 10) / 10).tolist()

# SLURM cluster config
SLURM_PROJECT_NAME = "dendrite"
SLURM_PARTITIONS = "partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100"
SLURM_CORES_PER_JOB = 48
# in GiB
_target_slurm_memory_per_job = 180
# get this by running fil-profile run debug.py (max memory used by a task)
# in practice, dask allocates ~55% of what is requested
_slurm_memory_per_task = 6
_slurm_threads_per_job = _target_slurm_memory_per_job / _slurm_memory_per_task
# https://github.com/dask/dask-jobqueue/issues/181#issue-372752428
SLURM_NUM_PROCESSES_PER_JOB = math.ceil(SLURM_CORES_PER_JOB / _slurm_threads_per_job)
SLURM_MEMORY_PER_JOB = math.ceil(_slurm_threads_per_job * _slurm_memory_per_task)
print(
    "Asking for {} cores, {} GiB of memory, {} processes per job".format(
        SLURM_CORES_PER_JOB, SLURM_MEMORY_PER_JOB, SLURM_NUM_PROCESSES_PER_JOB
    )
)
SLURM_WALLTIME = "120:00:00"
SLURM_MIN_JOBS = 20
SLURM_LOCAL_DIRECTORY = "/scratch/adhinart"
SLURM_SCALE_INTERVAL = "10s"  # how long until new scaling
SLURM_TARGET_DURATION = "60s"  # how long one task is supposed to take
SLURM_DASHBOARD_PORT = 8888
SLURM_WAIT_COUNT = 10
