import os
import numpy as np
import contextlib

from dataloader import FreSegDataset
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Optional, Callable

from utils import get_conf


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
    seed: int,
    num_threads: int,
):

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
            fold=-1,
            is_train=True, # ignored
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
    seed: int,
    n_jobs: int,
    num_threads: int,
):
    dataset = get_dataset(path_length, num_points, seed, num_threads)
    np.savez(
        os.path.join(output_dir, "spanning_paths.npz"),
        spanning_paths=dataset.spanning_paths,
    )

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
    def __init__(
        self,
        output_path: str,
        num_points: int,
        folds: List[List[int]],
        fold: int,
        is_train: bool,
        transform: Optional[Callable] = None,
    ):
        self.num_points = num_points
        self.transform = transform
        self.spanning_paths = np.load(
            os.path.join(output_path, "spanning_paths.npz"), allow_pickle=True
        )["spanning_paths"].item()

        if fold == -1:
            print("Loading all folds, ignoring is_train")
            trunk_ids = self.spanning_paths.keys()
        else:
            if is_train:
                trunk_ids = [
                    item
                    for idx, sublist in enumerate(folds)
                    if idx != fold
                    for item in sublist
                ]
            else:
                trunk_ids = folds[fold]
        self.trunk_ids = sorted(trunk_ids)

        files = []
        i = 0
        for id in sorted(self.spanning_paths.keys()):
            for path in self.spanning_paths[id]:
                if id in self.trunk_ids:
                    files.append(os.path.join(output_path, f"{i}.npz"))
                    assert os.path.exists(files[-1])
                i += 1
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        trunk_id, pc, trunk_pc, label = (
            data["trunk_id"],
            data["pc"],
            data["trunk_pc"],
            data["label"],
        )
        assert trunk_id in self.trunk_ids

        # PC is [N, 3], downsample to [num_points, 3]
        pc = np.random.permutation(pc)[: self.num_points]

        if self.transform is None:
            return trunk_id, pc, trunk_pc, label
        else:
            return self.transform(trunk_id, pc, trunk_pc, label)


if __name__ == "__main__":
    conf = get_conf()
    for configuration in conf.dataloader.cached:
        print(f"Caching dataset for {configuration}")
        if not os.path.exists(configuration.output_dir):
            os.makedirs(configuration.output_dir)

        cache_dataset(
            output_dir=configuration.output_dir,
            path_length=configuration.path_length,
            num_points=configuration.num_points,
            seed=0, # Assuming a default value for seed
            n_jobs=conf.n_jobs_cache,
            num_threads=conf.n_threads_cache
        )
