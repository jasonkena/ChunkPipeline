import numpy as np
import math

# set in ~/.config/dask
# DASK_CONFIG = {
# "array.slicing.split_large_chunks": False,
# }  # prevent GIL holding processes from crashing workers

# "distributed.scheduler.worker-ttl": None,
# "distributed.comm.timeouts.connect": "300s",
# worker-ttl:
# dask.config.set({"admin.tick.limit": "1h"})
# dask.config.set({"distributed.comm.retry.count": 3})
# dask.config.set({'distributed.scheduler.idle-timeout' : "5 minutes"})

# dask.config.set({"distributed.scheduler.worker-saturation": 1.0})

# __ to delimit hierarchy, read config.py
TASK = None  # den_seg/mouse/human/etc. this als

# {TASK} will be filled in at runtime
_base_path = "/mmfs1/data/adhinart/dendrite/data"
MISC__ENABLE_MEMUSAGE = False
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
    # "parallel": 0,  # use all cpu # raises ValueError('signal only works in main thread of the main interpreter')
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

# anisotropy (1, 1, 1)
VESSEL__MAX_ERODE = 5
VESSEL__ERODE_DELTA = 5
VESSEL__CONNECTIVITY = BASELINE__CONNECTIVITY

# SLURM cluster config
SLURM__PROJECT_NAME = "{TASK}"
# SLURM__PARTITIONS = "full_nodes48,full_nodes64,gpuv100,gpua100,weidf"
SLURM__PARTITIONS = "weidf,gpua100,gpuv100,exclusive"  # for some reason, if shared is included, it overrides everything else
# SLURM__PARTITIONS = "shared"
# SLURM__PARTITIONS = "partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100,weidf"
SLURM__CORES_PER_JOB = 48

# https://github.com/dask/dask-jobqueue/issues/181#issue-372752428
# multiprocess to release GIL
SLURM__NUM_PROCESSES_PER_JOB = 1
# SLURM__NUM_PROCESSES_PER_JOB = 3
# in GiB
SLURM__MEMORY_PER_JOB = 170  # used only to compute number of threads
# get this by running fil-profile run debug.py (max memory used by a task)
SLURM__MEMORY_PER_TASK = 20  # to calculate number of threads per job

# in hours
SLURM__WALLTIME = 2
SLURM__MIN_JOBS = 1
SLURM__MAX_JOBS = 20
SLURM__ADAPT_INTERVAL = "10s"  # wait 10 seconds before scaling
# Local directory has to be unique for each job
# random hex is to guarantee unique directory name
# SLURM__LOCAL_DIRECTORY = "/tmp/chunk_pipeline/$(openssl rand -hex 5)" # tmp supposedly gets cleared
SLURM__DELETE_LOCAL_DIRECTORY = "/scratch/adhinart/chunk_pipeline"
SLURM__LOCAL_DIRECTORY = (
    "/scratch/adhinart/chunk_pipeline/$SLURM_JOB_ID"  # scratch locks NFS
)
# local does not always have enough storage
# SLURM__LOCAL_DIRECTORY = "/local/adhinart/chunk_pipeline"  # assuming only a single job is placed on each node
SLURM__DASHBOARD_PORT = 8989
SLURM__LOG_DIRECTORY = "/mmfs1/data/adhinart/dendrite/logs"  # assuming only a single job is placed on each node
SLURM__INTERFACE = "ib0"

# low res foundation datasets
FOUNDATION__CHUNK_SIZE = (
    10,
    512,
    512,
)  # needs to be 10, otherwise it ruins mean and std calculations?
FOUNDATION__IGNORE_BELOW_STD = 0.0
FOUNDATION__GAUSSIAN_SIGMA = 1.0
FOUNDATION__THRESHOLD_Z_SCORE = 3.0  # require score to be above mean + z_score * std
FOUNDATION__DUST_THRESHOLD = 1000
FOUNDATION__DUST_CONNECTIVITY = 6
FOUNDATION__CONNECTIVITY = BASELINE__CONNECTIVITY

NIB = None  # form {"name": filename}
# junest cannot read /mmfs1 prefix
l1_path = "/data/adhinart/L1-Skeleton"
L1__BIN_PATH = f"junest {l1_path}/PointCloud/PointCloudL1"
# L1__JSON_PATH = f"{l1_path}/default_skeleton_config.json"
# L1__JSON_PATH = f"{l1_path}/default_skeleton_config.json"
L1__JSON_PATH = "/data/adhinart/dendrite/chunk_pipeline/configs/dendrite_skeleton_config.json"  # changed downsample num to 3000

# junest can't read /scratch
# NOTE: remember to clear this
L1__TMP_DIR = f"{l1_path}/tmp"
L1__STORE_TMP = True
# noise in isotropic nm space (pre downscaling)
L1__NOISE_STD = 0.0  # determine this by eyeballing downsampled pointcloud
# L1__NOISE_STD = 20.0 # determine this by eyeballing downsampled pointcloud
# L1__DOWNSCALE_FACTOR = 0.001
# set to 0 to use all points
L1__NUM_SAMPLE = 100000  # GUI example has 40000 points
L1__MAX_ERRORS = 5
L1__ERROR_UPSAMPLE = 1.5

VESICLE__GLOB = ["/mmfs1/data/adhinart/vesicle/new_xiaomeng/*/*.tif"]  # ["*/*"]
# VESICLE__GLOB = ["/mmfs1/data/adhinart/vesicle/new_im_vesicle/*.tif"]
VESICLE__BIN_PATH = "/mmfs1/data/adhinart/vesicle/run.sh"
# randomly wait X seconds before starting jobs
VESICLE__STAGGER = 10

# number of points in centerline interpolation
FRENET__PATH_LENGTH = 3000
# segment length in real units (taking into account anisotropy)
# FRENET__WINDOW_LENGTH = 500.0
# # stride length in real units (taking into account anisotropy)
# FRENET__STRIDE_LENGTH = 500.0

FRENET__WINDOW_LENGTH = 2000.0
FRENET__STRIDE_LENGTH = 1000.0
