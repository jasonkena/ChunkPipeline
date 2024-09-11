import os
import numpy as np
import contextlib

from dataloader import FreSegDataset
from tqdm import tqdm
from joblib import Parallel, delayed

import argparse


@contextlib.contextmanager
def temp_seed(seed):
    # https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_dataset(
    path_length: float,
    num_points: int,
    fold: int,
    is_train: bool,
    seed: int,
    num_threads: int,
):
    assert fold in [0, 1, 2, 3, 4]

    with temp_seed(seed):
        dataset = FreSegDataset(
            mapping_path="/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/mapping.npy",
            seed_path="/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/seed.npz",
            pc_zarr_path="/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/pc.zarr",
            pc_lengths_path="/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/pc_lengths.npz",
            path_length=path_length,
            num_points=num_points,
            anisotropy=[30, 6, 6],
            folds=[
                [3, 5, 11, 12, 23, 28, 29, 32, 39, 42],
                [8, 15, 19, 27, 30, 34, 35, 36, 46, 49],
                [9, 14, 16, 17, 21, 26, 31, 33, 43, 44],
                [2, 6, 7, 13, 18, 24, 25, 38, 41, 50],
                [1, 4, 10, 20, 22, 37, 40, 45, 47, 48],
            ],
            fold=fold,
            is_train=is_train,
            transform=None,
            num_threads=num_threads,
        )

    return dataset


def store_sample(dataset, i, output_dir):
    trunk_id, pc, trunk_pc, label = dataset[i]
    np.savez(
        os.path.join(output_dir, f"{i}.npz"),
        trunk_id=trunk_id,
        pc=pc,
        trunk_pc=trunk_pc,
        label=label,
    )


def cache_dataset(
    output_dir: str,
    path_length: float,
    num_points: int,
    fold: int,
    is_train: bool,
    seed: int,
    n_jobs: int,
    num_threads: int,
):
    dataset = get_dataset(path_length, num_points, fold, is_train, seed, num_threads)

    results = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(store_sample)(dataset, i, output_dir)
                for i in range(len(dataset))
            ),
            total=len(dataset),
            leave=False,
        )
    )

    return dataset


class CachedDataset:
    def __init__(self, output_path: str):
        self.output_path = output_path
        files = os.listdir(output_path)
        self.files = sorted(files, key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.output_path, self.files[idx]))
        return data["trunk_id"], data["pc"], data["trunk_pc"], data["label"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    parser.add_argument("--pathlength", type=float, required=True, help="Path length")
    parser.add_argument("--npoints", type=int, required=True, help="Number of points")
    parser.add_argument("--n_jobs", type=int, default=64, help="Number of jobs")
    # parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    args = parser.parse_args()

    output_dir = os.path.join(
        args.output_dir, f"dataset_{args.fold}_{args.pathlength}_{args.npoints}"
    )
    print(f"Output path: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cache_dataset(
        output_dir=output_dir,
        path_length=args.pathlength,
        num_points=args.npoints,
        fold=args.fold,
        is_train=True,  # Assuming a default value for is_train
        seed=0,  # Assuming a default value for seed
        n_jobs=args.n_jobs,
        num_threads=args.num_threads,
    )
# python cache_dataloader.py --fold 0 --pathlength 10000 --npoints 4096 --output_dir /data/adhinart/dendrite/scripts/igneous/outputs/seg_den
