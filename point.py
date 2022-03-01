import numpy as np
import chunk
import chunk_sphere
import h5py
import os
import sys
from settings import *
from utils import extend_bbox


def chunk_func_spine(params, all, spine):
    shrink_slices = params["shrink_slices"]
    assert shrink_slices is not None
    boundary = chunk_sphere._chunk_get_boundary(all)[0]
    return boundary[shrink_slices], spine[shrink_slices]


def main(input_path, id):
    # id is in range(50)
    all = h5py.File(os.path.join(input_path, f"{str(id)}.h5"))
    spine = h5py.File(os.path.join(input_path, "seg_den_spine_6nm.h5"))
    cache = h5py.File(os.path.join(input_path, "cache.h5"), "w")

    bboxes = np.load(os.path.join(input_path, "den_6nm_bb.npy")).astype(int)
    row = extend_bbox(bboxes[id - 1], spine.get("main").shape)

    output_file = os.path.join("results", f"{row[0]}.npy")
    if os.path.exists(output_file):
        return

    new_spine = cache.create_dataset(
        "new_spine",
        shape=(row[2] - row[1] + 1, row[4] - row[3] + 1, row[6] - row[5] + 1),
        dtype="uint16",
    )

    new_spine = chunk.get_seg(
        new_spine, spine.get("main"), row, CHUNK_SIZE, True, NUM_WORKERS
    )

    # NOTE: NOTE: NOTE: rewrite this to use get_seg
    output = chunk.chunk_argwhere(
        [all.get("main"), new_spine],
        CHUNK_SIZE,
        lambda params, all, spine: chunk_func_spine(params, all, spine),
        "extend",
        NUM_WORKERS,
    )
    output[:, :3] += np.array([row[1], row[3], row[5]])
    assert (0 <= np.min(output)) and (np.max(output) <= np.iinfo(np.uint16).max)
    output = output.astype(np.uint16)
    np.save(output_file, output)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
