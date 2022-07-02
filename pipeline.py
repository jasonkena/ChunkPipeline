import dask
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

from utils import dask_read_array, dask_write_array
from settings import *

import time

cluster = SLURMCluster(
    job_name=SLURM_PROJECT_NAME,
    queue=SLURM_PARTITIONS,
    cores=SLURM_CORES_PER_JOB,
    memory=SLURM_MEMORY_PER_JOB,
    # interface=SLURM_INTERFACE,
    # scheduler_options={"host":"localhost"},
    # extra=SLURM_EXTRA,
)
cluster.adapt(maximum_jobs=SLURM_MAX_JOBS)

client = Client(cluster)


@dask.delayed
def task():
    print("hi")
    return 1


tasks = [task() for _ in range(10)]
print(client)
a = dask.compute(*tasks)
print(a)
# client.shutdown()
