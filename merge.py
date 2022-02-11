import h5py
import numpy as np
from tqdm import tqdm

import os
import json


def merge(h5, bbox):
    shape = h5py.File(h5).get("main").shape
    result = np.zeros(shape, np.uint16)
    bbox = np.load(bbox)
    npys = sorted(os.listdir("results"))
    assert len(npys) == len(bbox)

    # first element in list always indicates trunk
    equivalency = {}
    last_id = 0
    for box in tqdm(bbox):
        subvol = np.load(os.path.join("results", f"{box[0]}.npy"))
        is_valid = subvol != 0
        max_id = np.max(subvol)

        equivalency[int(box[0])] = list(range(last_id + 1, last_id + max_id + 1))
        result[box[1] : box[2] + 1, box[3] : box[4] + 1, box[5] : box[6] + 1][
            is_valid
        ] = (subvol[is_valid] + last_id)
        last_id = last_id + max_id

    return result, equivalency


if __name__ == "__main__":
    result, equivalency = merge("./den_ruilin_v2_16nm.h5", "bbox.npy")
    np.save("merged.npy", result)
    json.dump(equivalency, open("equivalency.json", 'w'))
