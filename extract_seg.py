import numpy as np
import h5py
import os
import sys
import chunk
from settings import *
from utils import extend_bbox, create_compressed


def main(base_path, id):
    bboxes = np.load(os.path.join(base_path, "bbox.npy")).astype(int)

    # id is in range(50)
    all = h5py.File(os.path.join(base_path, "raw.h5"))
    row = extend_bbox(bboxes[id - 1], all.get("main").shape)

    extracted = h5py.File(os.path.join(base_path, f"{row[0]}.h5"), "w")

    main = create_compressed(
        extracted,
        "main",
        shape=(row[2] - row[1] + 1, row[4] - row[3] + 1, row[6] - row[5] + 1),
        dtype=bool,
    )
    create_compressed(extracted, "row", data=row)

    chunk.get_seg(main, all.get("main"), row, CHUNK_SIZE, True, NUM_WORKERS)

    extracted.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
