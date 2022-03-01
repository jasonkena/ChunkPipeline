import numpy as np
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from settings import *

import os
import sys
import json

import chunk


def main(output_path, inputs, idx):
    for i in idx:
        assert i <= NUM_DENDRITES

    final_output = h5py.File(os.path.join(output_path, "final.h5"), "w")
    bboxes = [
        h5py.File(os.path.join("extracted", f"{idx[i]}.h5")).get("row")
        for i in range(len(inputs))
    ]
    final = final_output.create_dataset(
        "main", shape=h5py.File("seg_den_6nm.h5").get("main").shape, dtype="uint16"
    )

    files = [h5py.File(i) for i in inputs]
    voxel_counts = [file.get("voxel_counts")[:] for file in files]

    # exclude 0 and 1, which represent background and trunk respectively
    num_spines = [i[2:].shape[0] for i in voxel_counts]
    # start indexing from
    cumsum_spines = np.cumsum([NUM_DENDRITES + 1] + num_spines)[:-1]
    remapping = {}
    for i in range(len(inputs)):
        remapping[idx[i]] = np.concatenate(
            [[0, idx[i]], np.arange(cumsum_spines[i], cumsum_spines[i] + num_spines[i])]
        )
    str_remapping = json.dumps({i: remapping[i].tolist() for i in remapping})
    final_output.attrs["remapping"] = str_remapping

    for i in range(len(inputs)):
        print(f"processing {files[i]}")
        chunk.merge_seg(
            final,
            files[i].get("seg"),
            bboxes[i],
            CHUNK_SIZE,
            lambda output, input: output + remapping[idx[i]][input],
            NUM_WORKERS,
        )
    final_output.close()


if __name__ == "__main__":
    inputs = sorted(glob("baseline/*"))
    idx = [int(Path(i).stem.split("_")[1]) for i in inputs]
    # list of ids to select
    filter = [14]
    if len(filter):
        inputs = [inputs[i] for i in range(len(idx)) if idx[i] in filter]
        idx = [idx[i] for i in range(len(idx)) if idx[i] in filter]
    main(sys.argv[1], inputs, idx)
