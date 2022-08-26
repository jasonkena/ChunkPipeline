import numpy as np
import math

DASK_CONFIG = {
    "array.slicing.split_large_chunks": False,
    "distributed.scheduler.worker-ttl": None,
}  # prevent GIL holding processes from crashing workers
# "distributed.comm.timeouts.connect": "300s",
# worker-ttl:
# dask.config.set({"admin.tick.limit": "1h"})
# dask.config.set({"distributed.comm.retry.count": 3})
# dask.config.set({'distributed.scheduler.idle-timeout' : "5 minutes"})

# __ to delimit hierarchy, read config.py
TASK = None  # den_seg/mouse/human/etc. this als

# {TASK} will be filled in at runtime
_base_path = "/mmfs1/data/adhinart/dendrite/data"
MISC__MEMUSAGE_PATH = _base_path + "/{TASK}/memusage.csv"
MISC__ZARR_PATH = _base_path + "/{TASK}/"


GENERAL__UINT_DTYPE = np.uint16
# INT_DTYPE = np.int16

# data storage will be at chunk size CHUNK_SIZE[i] // 2
# CHUNK_SIZE = "auto" # let dask choose chunk_size
GENERAL__CHUNK_SIZE = (512, 512, 512)
GENERAL__ANISOTROPY = None  # depends on the dataset, must be specified in config file
# for den_seg
# ANISOTROPY = (30, 6, 6)
# for human and mouse
# ANISOTROPY = (30, 8, 8)

H5 = None  # form {"raw": (file, dataset)}

# baseline related hyperparameters
BASELINE__CONNECTIVITY = 26
BASELINE__MAX_ERODE = 100
BASELINE__ERODE_DELTA = 100
BASELINE__NUM_ITER = 1

# point cloud inference hyperparameters
PC__DOWNSAMPLE_RADIUS = 200  # for expand_parabola threshold estimation
PC__PRED_THRESHOLD = 0.5
PC__TRUNK_RADIUS_DELTA = 30

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
# fuse: merge until only one connected component remaining
# list because config does not allow None values
KIMI__POSTPROCESS_PARAMS = {
    "dust_threshold": KIMI__PARAMS["dust_threshold"],
    "tick_threshold": 3500,
    "fuse_radius": 200,  # at what distance nodes should be fused
}
# where 6 represents the minimum anisotropy
assert KIMI__POSTPROCESS_PARAMS["fuse_radius"] < 6 * min(GENERAL__CHUNK_SIZE)

# evaluation hyperparameters
EVAL__THRESHOLD = (np.arange(1, 10) / 10).tolist()

# coarse blood segmentation hyperparameters
COARSE_ORIGINAL__IMAGE_PATH = (
    "/mmfs1/data/bccv/dataset/R0/im_64nm/*.png"  # glob pattern
)
COARSE_ORIGINAL__APPLY_CLAHE = True
COARSE_ORIGINAL__CLIP_LIMIT = 2.0
COARSE_ORIGINAL__TILE_GRID_SIZE = (8, 8)

COARSE__EROSION_STRUCTURE = np.ones([3, 3, 3], dtype=bool).tolist()
COARSE__DUST_THRESHOLD = 100
COARSE__DUST_CONNECTIVITY = 6
COARSE__CONNECTIVITY = BASELINE__CONNECTIVITY  # for connected components
COARSE__THRESHOLD_Z_SCORE = 1.0  # require score to be above mean + z_score * std

# SLURM cluster config
SLURM__PROJECT_NAME = "{TASK}"
SLURM__PARTITIONS = "partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100"
SLURM__CORES_PER_JOB = 48

# in GiB
SLURM__MEMORY_PER_JOB = 240
# get this by running fil-profile run debug.py (max memory used by a task)
SLURM__MEMORY_PER_TASK = 8  # to calculate number of threads per job
# https://github.com/dask/dask-jobqueue/issues/181#issue-372752428
# multiprocess to release GIL
SLURM__NUM_PROCESSES_PER_JOB = SLURM__MEMORY_PER_JOB // SLURM__MEMORY_PER_TASK
SLURM__WALLTIME = "120:00:00"
SLURM__MIN_JOBS = 20
# Local directory has to be unique for each job
# random hex is to guarantee unique directory name
# SLURM__LOCAL_DIRECTORY = "/tmp/chunk_pipeline/$(openssl rand -hex 5)" # tmp supposedly gets cleared
SLURM__LOCAL_DIRECTORY = (
    "/scratch/adhinart/chunk_pipeline/$SLURM_JOB_ID"  # scratch locks NFS
)
# SLURM__LOCAL_DIRECTORY = "/local/adhinart/chunk_pipeline"  # assuming only a single job is placed on each node
SLURM__DASHBOARD_PORT = 8888
SLURM__INTERFACE = "ib0"
