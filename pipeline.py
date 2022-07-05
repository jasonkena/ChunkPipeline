import dask
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster

import chunk
import sphere
from utils import dask_read_array, dask_write_array
from settings import *

if __name__ == "__main__":
    # with LocalCluster() as cluster, Client(cluster) as client:
    with SLURMCluster(
        local_directory=SLURM_LOCAL_DIRECTORY,
        job_name=SLURM_PROJECT_NAME,
        queue=SLURM_PARTITIONS,
        cores=SLURM_CORES_PER_JOB,
        memory=SLURM_MEMORY_PER_JOB,
        scheduler_options={"dashboard_address": f":{SLURM_DASHBOARD_PORT}"},
        walltime=SLURM_WALLTIME,
    ) as cluster, Client(cluster) as client:

        # print(cluster.job_script())
        # cluster.scale(5)
        # cluster.scale(jobs=SLURM_NUM_JOBS)
        cluster.adapt(
            minimum_jobs=SLURM_MIN_JOBS,
            maximum_jobs=SLURM_MAX_JOBS,
            interval=SLURM_SCALE_INTERVAL,
            waitcount=SLURM_WAIT_COUNT,
            target_duration=SLURM_TARGET_DURATION,
        )

        client = Client(cluster)
        print(client)

        import time

        shape = (1000, 1000, 1000)
        # chunk_size = (9, 8, 7)
        chunk_size = (10, 10, 10)

        input = np.random.randint(0, 2 ** 16 - 1, shape)
        print("done generating")
        start = time.time()
        output = chunk.chunk_bbox(da.from_array(input, chunks=chunk_size)).compute()
        end = time.time()
        print(end - start)
        __import__("pdb").set_trace()

import warnings

# ignore all warnings after execution
warnings.filterwarnings("ignore")
