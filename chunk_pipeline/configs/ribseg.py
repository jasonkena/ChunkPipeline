import glob

TASK = "ribseg"

data_path = "/mmfs1/data/adhinart/L1-Skeleton/seg"
NIB = {}

for file in sorted(glob.glob(data_path + "/*.nii.gz")):
    name = file.split("/")[-1].split(".")[0].split("-")[0]
    NIB[name] = file

SLURM__NUM_PROCESSES_PER_JOB = 20
SLURM__MEMORY_PER_TASK = 5
