import zarr
import glob
import numpy as np

from tqdm import tqdm

import os
import dask.distributed

from dask.diagnostics import ProgressBar

output_dir = "/mmfs1/scratch/adhinart/segment_export"


def get_tasks():
    paths = [
        "/mmfs1/data/adhinart/dendrite/data/seg_den",
        "/mmfs1/data/adhinart/dendrite/data/mouse",
        "/mmfs1/data/adhinart/dendrite/data/human",
    ]

    tasks = []
    for path in paths:
        dataset = path.split("/")[-1]
        for seg in tqdm(sorted(glob.glob(path + "/" + "point_cloud_segments_*"))):
            den_idx = seg.split("_")[-1]

            # for each dendrite
            group = zarr.open_group(seg)
            attrs = group["_attrs"][0]
            skel = attrs["skel"]
            skel_gnb = attrs["skel_gnb"]
            split_idx = attrs["split_idx"]

            keys = list(group.keys())
            keys = [k for k in keys if k[0] != "_"]
            keys = [k for k in keys if "gnb" not in k]
            idx = [int(k.split("_")[-1]) for k in keys]
            idx = sorted(idx)
            n = len(idx)
            assert idx == list(range(n))

            seen = []
            for i in tqdm(range(n)):
                if split_idx[i] in seen:
                    continue
                seen.append(split_idx[i])
                tasks.append((seg, dataset, den_idx, i, skel, skel_gnb, split_idx[i]))
    return tasks


def process_task(task):
    seg, dataset, den_idx, i, skel, skel_gnb, split_idx = task
    with zarr.open_group(seg) as group:
        pc = group[f"pc_{i}"][:]
        pc_gnb = group[f"pc_gnb_{i}"][:]

        np.savez_compressed(
            os.path.join(output_dir, f"{dataset}_{den_idx}_{i}.npz"),
            pc=pc,
            pc_gnb=pc_gnb,
            skel=skel,
            skel_gnb=skel_gnb,
            split_idx=split_idx,
        )
    return True


def dask_run():
    os.system("rm -rf {}".format(output_dir))
    # make dir
    os.makedirs(output_dir)

    print("deleted")
    client = dask.distributed.Client(n_workers=10, threads_per_worker=1)
    print(client)

    tasks = get_tasks()
    print(f"Total tasks: {len(tasks)}")
    tasks = [dask.delayed(process_task)(task) for task in tasks]

    dask.compute(tasks)
    print("done")


if __name__ == "__main__":
    # old_patch_output()

    dask_run()
