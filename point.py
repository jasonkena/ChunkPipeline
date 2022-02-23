import numpy as np
import chunk
import chunk_sphere
import h5py
import os
from settings import *


def _chunk_argwhere_seg(z, y, x, chunk_size, vol, id):
    # returns indices where vol is true
    idx = np.argwhere(vol == id)
    idx[:, 0] += chunk_size[0] * z
    idx[:, 1] += chunk_size[1] * y
    idx[:, 2] += chunk_size[2] * x

    return idx


def chunk_argwhere_seg(vol, chunk_size, bbox, num_workers):
    # TODO: implement chunked saving instead of aggregating all indices
    return np.concatenate(
        chunk.simple_chunk(
            None,
            [vol],
            chunk_size,
            _chunk_argwhere_seg,
            num_workers,
            pass_params=True,
            bbox=bbox,
            id=bbox[0],
        ).reshape(-1)
    )


if __name__ == "__main__":
    all = h5py.File("seg_den_6nm.h5")
    spine = h5py.File("seg_den_spine_6nm.h5")

    bboxes = np.loadtxt("den_6nm_bb.txt", dtype=int)
    output = h5py.File("test_point.h5", "w")

    for row in bboxes:
        output.create_dataset(str(row[0]), result.shape, dtype="uint16")[
            :
        ] = chunk_argwhere_seg(all.get("main"), CHUNK_SIZE, row, NUM_WORKERS)
