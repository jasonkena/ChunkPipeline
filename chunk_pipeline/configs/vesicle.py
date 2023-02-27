TASK = "vesicle"

SLURM__PARTITIONS = "gpuv100,gpua100"
# only run 8 instances of cellpose
SLURM__NUM_PROCESSES_PER_JOB = 8
SLURM__MEMORY_PER_TASK = 20  # only run 8 instances of cellpose?
