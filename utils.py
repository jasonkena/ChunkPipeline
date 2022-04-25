import numpy as np
import torch
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


def create_compressed(dataset, shape=None, *args, **kwargs):
    chunks = None
    if shape is not None:
        if len(shape) == 3:
            if not any([shape[i] < CHUNK_SIZE[i] for i in range(3)]):
                chunks = CHUNK_SIZE

    kwargs["shape"] = shape

    if chunks is not None:
        return dataset.create_dataset(
            *args, **kwargs, compression="gzip", chunks=chunks
        )
    else:
        return dataset.create_dataset(*args, **kwargs)
