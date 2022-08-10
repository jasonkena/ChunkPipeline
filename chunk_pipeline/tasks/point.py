import numpy as np
import dask
import dask.array as da
import chunk
import sphere
import h5py
import os
import sys
from settings import *
from utils import extend_bbox, dask_read_array
from dask.diagnostics import ProgressBar


def main(base_path, id):
    # id is in range(50)
    all = h5py.File(os.path.join(base_path, f"{str(id)}.h5")).get("main")
    all = dask_read_array(all)
    spine = h5py.File(os.path.join(base_path, "spine.h5")).get("main")
    spine = dask_read_array(spine)

    bboxes = np.load(os.path.join(base_path, "bbox.npy"))
    row = extend_bbox(bboxes[id - 1], spine.shape)

    sparse_file = os.path.join(base_path, f"sparse_{row[0]}.npy")
    dense_file = os.path.join(base_path, f"dense_{row[0]}.npy")
    if os.path.exists(sparse_file) and os.path.exists(dense_file):
        return

    new_spine = chunk.get_seg(spine, row, filter_id=True)

    if not os.path.exists(sparse_file):
        boundary = sphere.get_boundary(all)
        # this is probably what crashed
        sparse_output = chunk.chunk_nonzero(boundary, extra=new_spine)
        sparse_output = sparse_output + np.array([row[1], row[3], row[5], 0]).reshape(
            1, -1
        )
        np.save(sparse_file, sparse_output.compute())

    if not os.path.exists(dense_file):
        dense_output = chunk.chunk_nonzero(all, extra=new_spine)
        dense_output = dense_output + np.array([row[1], row[3], row[5], 0]).reshape(
            1, -1
        )
        np.save(dense_file, dense_output.compute())


if __name__ == "__main__":
    with ProgressBar():
        main(sys.argv[1], int(sys.argv[2]))
