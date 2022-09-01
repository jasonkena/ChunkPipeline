import h5py
import numpy as np

import dask
import dask.array as da


def task_load_h5(cfg):
    # will read raw , spine, and seg datasets
    general = cfg["GENERAL"]
    h5 = cfg["H5"]  # of form {"raw": (file, dataset)}

    result = {}
    for key, (file, dataset) in h5.items():
        dataset = h5py.File(file, "r").get(dataset)
        # first use chunks from original h5py (for fast chunk loading), then rechunk to match desired chunks
        result[key] = da.from_array(dataset, chunks=dataset.chunks).rechunk(
            general["CHUNK_SIZE"]
        )
    return result
