import numpy as np
import h5py
from utils import create_compressed, extend_bbox
from settings import *
import os
import math

import dask_chunk
import dask_chunk_sphere

import dask
import dask.array as da


def generate_batches(dataset, seeds, radius, num_points, anisotropy):
    # dataset [N, 4]
    # seeds [N, 4]
    new_dataset = dataset.copy()
    seeds = seeds.copy()

    # get real coordinates
    new_dataset[:, 1:4] = new_dataset[:, 1:4] * np.array(anisotropy).reshape(1, -1)
    seeds[:, 1:4] = seeds[:, 1:4] * np.array(anisotropy).reshape(1, -1)

    # [seeds, points, 3]
    delta = np.expand_dims(new_dataset[:, 1:4], 0) - np.expand_dims(seeds[:, 1:4], 1)
    # [seeds, points]
    delta = np.sum(delta ** 2, axis=-1)
    is_valid = delta < radius ** 2

    batch = []
    for i in range(seeds.shape[0]):
        idx = np.random.choice(np.where(is_valid[i])[0], size=num_points)
        # [N, 4]
        batch.append(dataset[idx])
    # [batch, N, num_points]
    batch = np.stack(batch, axis=0)

    # [points]
    in_range = np.any(is_valid, axis=0)

    return in_range, batch


def single_inference(
    tally, dataset, predict, radius, num_points, batch_size, anisotropy
):
    pool = dataset.copy()
    while pool.shape[0]:
        print(pool.shape[0])
        seed_idx = np.random.choice(pool.shape[0], size=min(pool.shape[0], batch_size))
        seeds = pool[seed_idx]  # [N, (idx, z,y,x)]
        # select points from the dataset and not the seed pool
        in_range, batch = generate_batches(
            dataset, seeds, radius, num_points, anisotropy
        )
        # [batch, N, 3], [batch, N]
        batch, idx = batch[..., 1:4], batch[..., 0]
        # [N, num_points]
        pred = predict(batch)

        for i in range(pred.shape[0]):
            # NOTE: if there are duplicate sampled points, the following will only run once per point
            # prediction, number of predictions
            tally[idx[i]] += np.vstack((pred[i], np.ones(pred[i].shape[0]))).T

        # remove points in range from the seed pool
        # since all idx in pool are in dataset
        mask = in_range[pool[:, 0]]
        pool = pool[~mask]

    return tally


def _chunk_max_pool(vol, block_info):
    return [np.max(vol)]


@dask.delayed
def _aggregate_dt(vol, real_anisotropy):
    dt = dask_chunk_sphere._get_dt(
        vol.astype(np.uint), real_anisotropy, black_border=False, block_info=None
    )
    return np.max(dt)


def max_dt(vol, chunk_width, anisotropy):
    # NOTE: NOTE: NOTE: implement this
    # chunk_width is in nanometers
    chunk_size = [math.ceil(chunk_width / anisotropy[i]) for i in range(3)]
    vol = da.rechunk(vol, chunks=tuple(chunk_size))

    downsampled = dask_chunk.chunk(_chunk_max_pool, [vol], [object])

    # an approximation of chunk_width
    real_anisotropy = [chunk_size[i] * anisotropy[i] for i in range(3)]
    max_dt_val = _aggregate_dt(downsampled, real_anisotropy)

    return max_dt_val


def inference(
    row,
    vol_dataset,
    raw_dataset,
    predict,
    elimination_radius,
    downsample_radius,
    num_points,
    batch_size,
    anisotropy,
    num_iter,
    threshold,
    chunk_size,
    connectivity,
    num_workers,
):
    # threshold is the the prediction threshold for something to be considered a spine as opposed to the trunk
    # downsample_radius is the width of a chunk to max pool to obtain max_dt

    # strip additional information
    raw_dataset = raw_dataset[:, :3]
    # where vol_dataset is an h5 dataset which contains volume of dendrite
    # assuming raw_dataset of form [N, 3]
    # where columns are Z, Y, X (not adjusted for anisotropy
    dataset = np.hstack((np.arange(raw_dataset.shape[0]).reshape(-1, 1), raw_dataset))

    # first column for average predictions, second column for number of predictions
    tally = np.zeros((dataset.shape[0], 2), dtype=np.float32)
    for _ in range(num_iter):
        tally = single_inference(
            tally,
            dataset,
            predict,
            elimination_radius,
            num_points,
            batch_size,
            anisotropy,
        )

    # only points which are actually sampled are valid
    is_sampled = tally[:, 1] > 0
    # 1 is trunk, 2 is spine, to be able to distinguish them from background
    pred = ((tally[is_sampled, 0] / tally[is_sampled, 1]) > threshold) + 1
    points = raw_dataset[is_sampled]

    # offset by bounding box
    points = points - np.array([row[1], row[3], row[5]]).reshape(1, -1)

    # now do volume filling
    shape = vol_dataset.shape
    seeded = chunk_seed(shape, points, pred, chunk_size)

    threshold = max_dt(vol_dataset, downsample_radius, anisotropy).compute()
    trunk_dt = dask_chunk_sphere.get_dt(
        seeded,
        anisotropy,
        False,
        threshold,
        filter_idx=1,
    )
    spine_dt = dask_chunk_sphere.get_dt(
        seeded,
        anisotropy,
        False,
        threshold,
        filter_idx=2,
    )
    # background 0, trunk 1, spine 2
    nearest_labels = ((spine_dt > trunk_dt) + 1) * vol_dataset

    final, voxel_counts = dask_chunk.chunk_cc3d(
        nearest_labels,
        connectivity,
        num_workers,
    )

    return final, voxel_counts


def dumb_predict(points):
    ALL_BATCHES.create_dataset(str(len(ALL_BATCHES.keys())), data=points)
    return np.random.rand(*points.shape[:2])
    # takes numpy array [N, num_points, 3]


def _chunk_seed(vol, merged, block_info):
    merged = merged.item()
    if merged is not None:
        vol = np.zeros_like(vol)
        merged = merged - np.array(
            [block_info[0]["array-location"][i][0] for i in range(3)] + [0]
        ).reshape(1, -1)
        vol[merged[:, 0], merged[:, 1], merged[:, 2]] = merged[:, 3]
    return [vol]


def chunk_seed(vol_shape, points, pred, chunk_size):
    chunk_idx = np.floor(
        points.astype(float) / np.array(chunk_size).reshape(1, -1)
    ).astype(int)
    # group points by chunk_idx
    inverse = np.unique(chunk_idx, return_inverse=True, axis=0)[1]
    argsort = np.argsort(inverse)

    chunk_idx = chunk_idx[argsort]
    merged = np.concatenate([points, pred.reshape(-1, 1)], axis=1)[argsort]

    # https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
    unique_idx, split_points = np.unique(chunk_idx, return_index=True, axis=0)
    # remove 0
    split_points = split_points[1:]
    splitted = np.split(merged, split_points, axis=0)
    chunked = np.empty(
        [math.ceil(vol_shape[i] / chunk_size[i]) for i in range(3)], dtype=object
    )

    for i, x in enumerate(unique_idx):
        chunked[x[0], x[1], x[2]] = splitted[i]

    result = dask_chunk.chunk(
        _chunk_seed,
        [
            da.zeros(vol_shape, chunks=chunk_size, dtype=int),
            da.from_array(chunked, chunks=(1, 1, 1)),
        ],
        [int],
    )

    return result


if __name__ == "__main__":
    ALL_BATCHES = h5py.File("dumb/batches.h5", "w")
    h5 = h5py.File("dumb/1.h5", "r")
    vol_dataset = h5.get("main")
    vol_dataset = da.from_array(vol_dataset, chunks=CHUNK_SIZE)
    row = h5.get("row")
    # points = np.load("dumb/1.npy")
    points = np.load("dumb/1.npy")[:30000]
    final = inference(
        row,
        vol_dataset,
        points,
        dumb_predict,
        PC_ELIMINATION_RADIUS,
        PC_DOWNSAMPLE_RADIUS,
        PC_NUM_POINTS,
        PC_BATCH_SIZE,
        ANISOTROPY,
        PC_NUM_ITER,
        PC_THRESHOLD,
        CHUNK_SIZE,
        CONNECTIVITY,
        NUM_WORKERS,
    )
