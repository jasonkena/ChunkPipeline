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

# SLURM cluster config
SLURM_PROJECT_NAME = "dendrite"
SLURM_PARTITIONS = "partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100"
SLURM_CORES_PER_JOB = 4
SLURM_MEMORY_PER_JOB = "4GB"
SLURM_WALLTIME = "120:00:00"
SLURM_MAX_JOBS = 100
