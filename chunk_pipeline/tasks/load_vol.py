import h5py
import numpy as np
import nibabel as nib

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


def task_load_nib(cfg):
    general = cfg["GENERAL"]
    nib_cfg = cfg["NIB"]  # of form {"name": filename}
    result = {}

    for key in nib_cfg:
        # load array proxy and not actual data
        # https://nipy.org/nibabel/nibabel_images.html#array-proxies
        dataset = nib.load(nib_cfg[key]).dataobj
        result[key] = da.from_array(dataset, chunks=general["CHUNK_SIZE"])

    return result
