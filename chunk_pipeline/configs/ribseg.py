import glob

TASK = "ribseg"

data_path = "/data/adhinart/ribseg/outputs/dgcnn_two_stage_preprocessed"
GENERAL__ANISOTROPY = (1, 1, 1)  # isotropic data

# set to 0 to use all points
L1__NOISE_STD = 0.0
# L1__NOISE_STD = 5.0
L1__NUM_SAMPLE = 50000
l1_path = "/data/adhinart/L1-Skeleton"
L1__JSON_PATH = f"{l1_path}/default_skeleton_config.json"
# L1__DOWNSCALE_FACTOR = 0.01

ids = list(range(1, 25))
seg_keys = ["gt", "pred"]
NPZ = {}

for file in sorted(glob.glob(data_path + "/*.npz")):
    file_name = file.split("/")[-1].split(".")[0].replace("RibFrac", "")
    for id in ids:
        temp_name = f"{file_name}_{id}"
        for seg_key in seg_keys:
            name = f"{temp_name}_{seg_key}"
            NPZ[name] = (file, id, "input", seg_key)

SLURM__NUM_PROCESSES_PER_JOB = 10
SLURM__MEMORY_PER_TASK = 10
