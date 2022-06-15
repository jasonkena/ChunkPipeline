import numpy as np
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from settings import *
from utils import create_compressed, dask_read_array, dask_write_array
import dask
import dask.array as da
import dask_chunk

import os
import sys
import json

import chunk


def main(base_path, inputs, idx):
    for i in idx:
        assert i <= NUM_DENDRITES

    output = os.path.join(base_path, "final.h5")
    final_output = h5py.File(output, "w")
    bboxes = [
        h5py.File(os.path.join(base_path, "extracted", f"{idx[i]}.h5")).get("row")
        for i in range(len(inputs))
    ]

    files = [h5py.File(i) for i in inputs]
    seg_bboxes = [file.get("seg_bbox")[:].copy() for file in files]
    for i in range(len(idx)):
        seg_bboxes[i][:, 1:3] += bboxes[i][1]
        seg_bboxes[i][:, 3:5] += bboxes[i][3]
        seg_bboxes[i][:, 5:7] += bboxes[i][5]

    # exclude 0 and 1, which represent background and trunk respectively
    num_spines = [i[1:].shape[0] for i in seg_bboxes]
    # start indexing from
    cumsum_spines = np.cumsum([NUM_DENDRITES + 1] + num_spines)[:-1]
    remapping = {}
    for i in range(len(inputs)):
        temp = np.concatenate(
            [[0, idx[i]], np.arange(cumsum_spines[i], cumsum_spines[i] + num_spines[i])]
        )
        # take into account background
        assert len(temp) == len(seg_bboxes[i]) + 1
        remapping[idx[i]] = temp
        seg_bboxes[i][:, 0] = temp[seg_bboxes[i][:, 0]]
    str_remapping = json.dumps({i: remapping[i].tolist() for i in remapping})
    final_output.attrs["remapping"] = str_remapping
    # unique to sort lexicographically
    final_output.create_dataset(
        "seg_bboxes", data=np.unique(np.concatenate(seg_bboxes, axis=0), axis=0)
    )
    final_output.close()

    final = da.zeros(
        shape=h5py.File(os.path.join(base_path, "raw.h5")).get("main").shape, dtype=int
    )

    for i in range(len(inputs)):
        input = dask_read_array(files[i].get("seg"))
        shape = input.shape
        # NOTE: since Dask doesn't allow multidimensional indexing
        remapped = da.from_array(remapping[idx[i]])[input.flatten()].reshape(shape)
        dask_chunk.merge_seg(
            final, remapped, bboxes[i], lambda output, input: output + input
        )

    dask_write_array(output, "main", final)


if __name__ == "__main__":
    inputs = sorted(glob(os.path.join(sys.argv[1], "baseline/*")))
    idx = [int(Path(i).stem.split("_")[1]) for i in inputs]
    # list of ids to select
    filter = []
    if len(filter):
        inputs = [inputs[i] for i in range(len(idx)) if idx[i] in filter]
        idx = [idx[i] for i in range(len(idx)) if idx[i] in filter]
    main(sys.argv[1], inputs, idx)
