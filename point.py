import numpy as np
import chunk
import chunk_sphere
import h5py
import os
from settings import *


def _chunk_argwhere(z, y, x, chunk_size, *args, **kwargs):
    # kwargs must contain func, and it must return 2 things: binary mask and another (array or None)
    # returns indices where vol is true
    new_kwargs = kwargs.copy()
    func = new_kwargs.pop("chunk_func")
    mask, extra = func(*args, **new_kwargs)

    idx = np.argwhere(mask)
    if extra is not None:
        extra_dim = 1 if mask.ndim < 4 else mask.shape[3]
        idx = np.concatenate((idx, extra[idx].reshape(-1, extra_dim)), axis=1)
    idx[:, 0] += chunk_size[0] * z
    idx[:, 1] += chunk_size[1] * y
    idx[:, 2] += chunk_size[2] * x

    return idx


def chunk_argwhere(dataset_inputs, chunk_size, chunk_func, bbox, num_workers):
    # TODO: implement chunked saving instead of aggregating all indices
    return np.concatenate(
        chunk.simple_chunk(
            None,
            dataset_inputs,
            chunk_size,
            _chunk_argwhere,
            num_workers,
            pass_params=True,
            bbox=bbox,
            chunk_func=chunk_func,
        ).reshape(-1)
    )


def chunk_func_spine(all, spine, id):
    return all == id, spine == id


if __name__ == "__main__":
    all = h5py.File("seg_den_6nm.h5")
    spine = h5py.File("seg_den_spine_6nm.h5")

    bboxes = np.loadtxt("den_6nm_bb.txt", dtype=int)
    output = h5py.File("test_point.h5", "w")

    for row in bboxes:
        output.create_dataset(
            str(row[0]),
            data=chunk_argwhere(
                [all.get("main"), spine.get("main")],
                CHUNK_SIZE,
                lambda all, spine: chunk_func_spine(all, spine, row[0]),
                row,
                NUM_WORKERS,
            ),
            dtype="uint16",
        )
