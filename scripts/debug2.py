import dask
import dask.array as da
import dask.bag as db
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from tqdm import tqdm

from dask_jobqueue import SLURMCluster

import config
from settings import *


@dask.delayed
def task():
    return config.STATE


if __name__ == "__main__":
    with SLURMCluster(
        local_directory=SLURM__LOCAL_DIRECTORY,
        job_name=SLURM__PROJECT_NAME,
        queue=SLURM__PARTITIONS,
        cores=SLURM__CORES_PER_JOB,
        memory=f"{SLURM__MEMORY_PER_JOB}GiB",
        scheduler_options={"dashboard_address": f":{SLURM__DASHBOARD_PORT}"},
        walltime=SLURM__WALLTIME,
        processes=SLURM__NUM_PROCESSES_PER_JOB,
        interface="ib0",
    ) as cluster, Client(cluster) as client:

        print(cluster.dashboard_link)
        cluster.scale(jobs=SLURM__MIN_JOBS)
        config.init()
        print(config.STATE)

        a = [task() for _ in range(10)]
        b = dask.compute(*a)
        __import__("pdb").set_trace()
