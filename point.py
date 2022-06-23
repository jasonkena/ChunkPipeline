import numpy as np
import dask
import dask.array as da
import dask_chunk
import dask_chunk_sphere
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

    bboxes = np.load(os.path.join(base_path, "bbox.npy")).astype(int)
    row = extend_bbox(bboxes[id - 1], spine.shape)

    output_file = os.path.join(base_path, f"{row[0]}.npy")
    if os.path.exists(output_file):
        return

    new_spine = dask_chunk.get_seg(spine, row, filter_id=True)

    boundary = dask_chunk_sphere.get_boundary(all)
    output = dask_chunk.chunk_nonzero(boundary, extra=new_spine)
    output = output + np.array([row[1], row[3], row[5], 0]).reshape(1, -1)

    np.save(output_file, output.compute())


if __name__ == "__main__":
    with ProgressBar():
        main(sys.argv[1], int(sys.argv[2]))
