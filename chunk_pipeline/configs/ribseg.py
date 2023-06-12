import glob

TASK = "ribseg"

data_path = "/mmfs1/data/adhinart/L1-Skeleton/seg"
NIB = {}

L1__IDS = list(range(1, 25))
# set to 0 to use all points
L1__NOISE_STD = 0.0
# L1__NOISE_STD = 5.0
L1__NUM_SAMPLE = 50000
l1_path = "/data/adhinart/L1-Skeleton"
L1__JSON_PATH = f"{l1_path}/default_skeleton_config.json"
# L1__DOWNSCALE_FACTOR = 0.01

special = [18, 20, 121, 122, 128, 131, 133, 137, 138, 146, 150, 151, 153, 155, 157, 159, 161, 164, 166, 168, 169, 172, 173, 175, 177, 179, 181, 182, 190, 191, 194, 197, 200, 203, 211, 220, 223, 227, 228, 229, 231]
special = [f"RibFrac{num}" for num in special]

for file in sorted(glob.glob(data_path + "/*.nii.gz")):
    name = file.split("/")[-1].split(".")[0].split("-")[0]
    if name in special:
        NIB[name] = file

SLURM__NUM_PROCESSES_PER_JOB = 5
SLURM__MEMORY_PER_TASK = 20
