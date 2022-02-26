import numpy as np
import chunk
import chunk_sphere
import h5py
import os
import sys
from settings import *

def chunk_func_spine(params, all, spine, id):
    shrink_slices = params["shrink_slices"]
    assert shrink_slices is not None
    boundary = chunk_sphere._chunk_get_boundary(all == id)[0]
    return boundary[shrink_slices], (spine == id)[shrink_slices]


def main(input_path, id):
    # id is in range(50)
    all = h5py.File(os.path.join(input_path, "seg_den_6nm.h5"))
    spine = h5py.File(os.path.join(input_path, "seg_den_spine_6nm.h5"))

    bboxes = np.loadtxt(os.path.join(input_path, "den_6nm_bb.txt"), dtype=int)
    row = bboxes[id]

    output_file = os.path.join("results", f"{row[0]}.npy")
    if os.path.exists(output_file):
        return

    output = chunk.chunk_argwhere(
        [all.get("main"), spine.get("main")],
        CHUNK_SIZE,
        lambda params, all, spine: chunk_func_spine(params, all, spine, row[0]),
        row,
        "extend",
        NUM_WORKERS,
    ).astype(np.uint16)
    assert (0 <= np.min(output)) and (np.max(output) <= np.iinfo(np.uint16))
    output = output.astype(np.uint16)
    np.save(output_file, output)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
