import os
import sys
import h5py
import numpy as np
import kimimaro

from skeleton import skel
from settings import *


def main(base_path, id):
    input = h5py.File(os.path.join(base_path, f"{str(id)}.h5"))
    row = input.get("row")[:]
    # read into memory, may need to refactor if out of memory
    input = input.get("main")[:]

    seg_skeleton = kimimaro.skeletonize(input, **KIMI_PARAMS)[1]
    # undo anisotropy
    seg_skeleton.vertices /= np.array(ANISOTROPY)
    # offset by bounding box
    seg_skeleton.vertices += np.array([row[1], row[3], row[5]]).reshape(1, 3)
    # choose random point to find furthest point, using it as seed
    seed = skel.find_furthest_pt(seg_skeleton, 0, single=False)[0]
    longest_path = skel.find_furthest_pt(seg_skeleton, seed, single=False)[1][0]

    output_path = os.path.join(base_path, f"skel_{str(id)}.h5")
    file = skel.write_skel(seg_skeleton, output_path)

    file.create_dataset("longest_path", data=longest_path)
    file.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
