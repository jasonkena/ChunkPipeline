import numpy as np
import h5py
import os
import sys
import chunk
from settings import *
from utils import extend_bbox, dask_read_array, dask_write_array
from dask.diagnostics import ProgressBar


def main(base_path, id):
    bboxes = np.load(os.path.join(base_path, "bbox.npy")).astype(int)

    # id is in range(50)
    all = h5py.File(os.path.join(base_path, "raw.h5")).get("main")
    all = dask_read_array(all)
    row = extend_bbox(bboxes[id - 1], all.shape)

    output_file = os.path.join(base_path, f"{row[0]}.h5")

    seg = chunk.get_seg(all, row, filter_id=True)

    file = dask_write_array(output_file, "main", seg)
    file.create_dataset("row", data=row)
    file.close()


if __name__ == "__main__":
    with ProgressBar():
        main(sys.argv[1], int(sys.argv[2]))
