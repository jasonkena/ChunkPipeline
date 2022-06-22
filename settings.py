# data storage will be at chunk size CHUNK_SIZE[i] // 2
# CHUNK_SIZE = "auto" # let dask choose chunk_size
CHUNK_SIZE = (500, 500, 500)
NUM_WORKERS = 20
ANISOTROPY = (30, 6, 6)

# how many attempts to load chunk
NUM_RETRY = 3

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
