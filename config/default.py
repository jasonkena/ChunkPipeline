import numpy as np
import math

GENERAL__UINT_DTYPE = np.uint16
# INT_DTYPE = np.int16

# data storage will be at chunk size CHUNK_SIZE[i] // 2
# CHUNK_SIZE = "auto" # let dask choose chunk_size
GENERAL__CHUNK_SIZE = (512, 512, 512)
GENERAL__ANISOTROPY = None # depends on the dataset, must be specified in config file
# for den_seg
# ANISOTROPY = (30, 6, 6)
# for human and mouse
# ANISOTROPY = (30, 8, 8)

# baseline related hyperparameters
BASELINE__CONNECTIVITY = 26
BASELINE__MAX_ERODE = 100
BASELINE__ERODE_DELTA = 100
BASELINE__NUM_ITER = 1

# point cloud inference hyperparameters
PC__DOWNSAMPLE_RADIUS = 200
PC__PRED_THRESHOLD = 0.5

# merging hyperparameters
# NUM_DENDRITES = 50

# skeletonization hyperparameters

KIMI__DOWNSAMPLE_RADIUS = 30
KIMI__PARAMS = {
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
EVAL__THRESHOLD = (np.arange(1, 10) / 10).tolist()

# SLURM cluster config
SLURM__PROJECT_NAME = "dendrite"
SLURM__PARTITIONS = "partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100"
SLURM__CORES_PER_JOB = 48

# in GiB
_target_slurm_memory_per_job = 180
# get this by running fil-profile run debug.py (max memory used by a task)
# in practice, dask allocates ~55% of what is requested
_slurm_memory_per_task = 6
_slurm_threads_per_job = _target_slurm_memory_per_job / _slurm_memory_per_task

# https://github.com/dask/dask-jobqueue/issues/181#issue-372752428
SLURM__NUM_PROCESSES_PER_JOB = math.ceil(SLURM__CORES_PER_JOB / _slurm_threads_per_job)
SLURM__MEMORY_PER_JOB = math.ceil(_slurm_threads_per_job * _slurm_memory_per_task)
SLURM__WALLTIME = "120:00:00"
SLURM__MIN_JOBS = 20
SLURM__LOCAL_DIRECTORY = "/scratch/adhinart"
SLURM__DASHBOARD_PORT = 8888
SLURM__INTERFACE = "ib0"

MISC__MEMUSAGE_PATH = "/mmfs1/data/adhinart/dendrite/memusage.csv"
MISC__ZARR_PATH = "/mmfs1/data/adhinart/dendrite/zarr"
MISC__CONFIG_PATH = "/mmfs1/data/adhinart/dendrite/config"

