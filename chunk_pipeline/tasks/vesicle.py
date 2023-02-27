import GPUtil
import numpy as np
import backoff
from time import sleep

from subprocess import Popen, PIPE
import os
import glob

import dask


def choose_gpu():
    gpus = GPUtil.getGPUs()
    percentages = {gpu.id: gpu.memoryFree / gpu.memoryTotal for gpu in gpus}
    # only keep gpus with more than 10% free memory
    percentages = {k: v for k, v in percentages.items() if v > 0.2}
    if len(percentages) == 0:
        raise Exception("No GPUs available")
    p = np.array(list(percentages.values()))
    p = np.exp(p)
    p = p / np.sum(p)
    gpu_id = np.random.choice(list(percentages.keys()), p=p)

    return gpu_id


@dask.delayed
@backoff.on_exception(backoff.expo, Exception, max_time=2000)
def run_cellpose(filename, cmd, stagger):
    # https://stackoverflow.com/questions/24849998/how-to-catch-exception-output-from-python-subprocess-check-output
    call = Popen([cmd, filename, str(choose_gpu())], stdout=PIPE, stderr=PIPE)
    output, error = call.communicate()
    if call.returncode != 0:
        raise Exception(f"cellpose failed on {filename}, {output}, {error}")

    return True


def task_run_vesicle(cfg):
    general = cfg["GENERAL"]
    vesicle = cfg["VESICLE"]

    filenames = [glob.glob(x) for x in vesicle["GLOB"]]
    filenames = [item for sublist in filenames for item in sublist]
    filenames = sorted(filenames)

    results = {
        filename: run_cellpose(filename, vesicle["BIN_PATH"], vesicle["STAGGER"])
        for filename in filenames
        if not os.path.exists(f"{filename.split('.')[0]}_seg.npy")
    }

    return results
