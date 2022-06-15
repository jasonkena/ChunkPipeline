import numpy as np
import dask
import dask.array as da
import h5py
from settings import *


def pad_vol(vol, kernel_shape):
    # given a 3d volume and a kernel shape, pad the input so that applying the kernel on the volume will result in a volume with the original shape
    # vol: 3d volume
    # kernel_shape: [z_size, y_size, x_size]
    assert np.all(np.array(kernel_shape) % 2 == 1)
    padded_vol = np.pad(
        vol,
        [
            [kernel_shape[0] // 2] * 2,
            [kernel_shape[1] // 2] * 2,
            [kernel_shape[2] // 2] * 2,
        ],
    )
    return padded_vol


def extend_bbox(bbox, max_shape):
    bbox = bbox.copy()
    bbox[1] = max(0, bbox[1] - 1)
    bbox[3] = max(0, bbox[3] - 1)
    bbox[5] = max(0, bbox[5] - 1)
    # -1 because of inclusive indexing
    bbox[2] = min(max_shape[0] - 1, bbox[2] + 1)
    bbox[4] = min(max_shape[1] - 1, bbox[4] + 1)
    bbox[6] = min(max_shape[2] - 1, bbox[6] + 1)

    return bbox


def create_compressed(dataset, name, shape, *args, **kwargs):
    assert len(shape) >= 3

    chunk_size = [i // 2 for i in CHUNK_SIZE]
    chunk_size = [min(chunk_size[i], shape[i]) for i in range(3)]
    chunk_size = tuple(chunk_size + list(shape[3:]))

    kwargs["name"] = name
    kwargs["shape"] = shape

    return dataset.create_dataset(
        *args, **kwargs, compression="gzip", chunks=chunk_size
    )


def dask_read_array(dataset):
    return da.from_array(dataset, chunks=CHUNK_SIZE)


def dask_write_array(filename, dataset_name, x):
    assert len(x.shape) >= 3
    # if list
    if not isinstance(dataset_name, list):
        dataset_name = [dataset_name]
    if not isinstance(x, list):
        x = [x]
    dataset_name = [f"/{i}" for i in dataset_name]

    # propagating chunk sizes
    da.to_hdf5(filename, dict(zip(dataset_name, x)), compression="gzip")
    return h5py.File(filename, "w")
