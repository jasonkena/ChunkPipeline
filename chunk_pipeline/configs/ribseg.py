import glob

TASK = "ribseg"

data_path = "/mmfs1/data/adhinart/L1-Skeleton/seg"
NIB = {}

L1__IDS = list(range(1, 25))
# set to 0 to use all points
L1__NOISE_STD = 5.0
L1__NUM_SAMPLE = 0
l1_path = "/data/adhinart/L1-Skeleton"
L1__JSON_PATH = f"{l1_path}/default_skeleton_config.json"
L1__DOWNSCALE_FACTOR = 0.01

for file in sorted(glob.glob(data_path + "/*.nii.gz")):
    name = file.split("/")[-1].split(".")[0].split("-")[0]
    NIB[name] = file

SLURM__NUM_PROCESSES_PER_JOB = 20
SLURM__MEMORY_PER_TASK = 5
