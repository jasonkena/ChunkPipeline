import numpy as np
import h5py
import os
import sys
import chunk
from settings import *
from utils import extend_bbox, dask_read_array, dask_write_array
from dask.diagnostics import ProgressBar


def main(base_path, id, h5):
    assert h5 in ["raw", "raw_gt"]
    bboxes = np.load(os.path.join(base_path, "bbox.npy")).astype(int)

    # id is in range(50)
    row = extend_bbox(bboxes[id - 1], all.shape)

    if "gt" not in h5:
        raw = h5py.File(os.path.join(base_path, "raw.h5")).get("main")
        raw = dask_read_array(raw)

        output_file = os.path.join(base_path, f"{row[0]}.h5")
        seg = chunk.get_seg(all, row, filter_id=True)
    else:
        raw = h5py.File(os.path.join(base_path, f"{row[0]}.h5")).get("main")
        raw = dask_read_array(raw)

        raw_gt = h5py.File(os.path.join(base_path, "raw_gt.h5")).get("main")
        raw_gt = dask_read_array(raw_gt)

        output_file = os.path.join(base_path, f"gt_{row[0]}.h5")
        seg = chunk.get_seg(raw_gt, row, filter_id=False) * raw

    file = dask_write_array(output_file, "main", seg)
    file.create_dataset("row", data=row)
    file.close()


if __name__ == "__main__":
    with ProgressBar():
        main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
