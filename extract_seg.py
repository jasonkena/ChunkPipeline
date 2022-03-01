import numpy as np
import h5py
import os
import sys
import chunk
from settings import *
from utils import extend_bbox


def main(input_path, id):
    bboxes = np.load(os.path.join(input_path, "den_6nm_bb.npy")).astype(int)

    # id is in range(50)
    all = h5py.File(os.path.join(input_path, "seg_den_6nm.h5"))
    row = extend_bbox(bboxes[id - 1], all.get("main").shape)

    output_file = os.path.join("extracted", f"{row[0]}.h5")
    if os.path.exists(output_file):
        return
    extracted = h5py.File(output_file, "w")

    main = extracted.create_dataset(
        "main",
        shape=(row[2] - row[1] + 1, row[4] - row[3] + 1, row[6] - row[5] + 1),
        dtype=bool,
    )
    extracted.create_dataset("row", data=row)

    chunk.get_seg(main, all.get("main"), row, CHUNK_SIZE, True, NUM_WORKERS)

    extracted.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
