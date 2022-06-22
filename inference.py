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
    merged_pred,
    downsample_radius,
    anisotropy,
    pred_threshold,
    chunk_size,
    connectivity,
):
    # merged_pred ([N, 4]) are the points in original coordinate system (pre-anisotropy and cropping), with a column for predictions
    # duplicate predictions are handled appropriately

    # threshold is the the prediction threshold for something to be considered a spine as opposed to the trunk
    # downsample_radius is the width of a chunk to max pool to obtain max_dt
    # where vol_dataset is an h5 dataset which contains volume of dendrite

    # first column for average predictions, second column for number of predictions

    points, pred = merged_pred[:, :3], merged_pred[:, 3]
    # NOTE: refactor; implement groupby function
    # deduplicate pred by averaging predictions
    inverse = np.unique(points, return_inverse=True, axis=0)[1]
    argsort = np.argsort(inverse)

    points = points[argsort]
    pred = pred[argsort]
    points, split_idx = np.unique(points, return_index=True, axis=0)
    # remove 0
    split_idx = split_idx[1:]
    pred = np.split(pred, split_idx)
    pred = np.array([np.mean(v) for v in pred]).reshape(-1, 1)

    # 1 for trunk, 2 for spine
    pred = (pred > pred_threshold) + 1

    # offset by bounding box
    points = points - np.array([row[1], row[3], row[5]]).reshape(1, -1)

    # now do volume filling
    shape = vol_dataset.shape
    seeded = chunk_seed(shape, points, pred, chunk_size)

    # NOTE: downsample_radius could be a hyperparameter; it is computed for correctness
    dt_threshold = max_dt(vol_dataset, downsample_radius, anisotropy).compute()
    trunk_dt = dask_chunk_sphere.get_dt(
        seeded,
        anisotropy,
        False,
        dt_threshold,
        filter_idx=1,
    )
    spine_dt = dask_chunk_sphere.get_dt(
        seeded,
        anisotropy,
        False,
        dt_threshold,
        filter_idx=2,
    )
    # background 0, trunk 1, spine 2
    nearest_labels = ((spine_dt < trunk_dt) + 1) * vol_dataset

    final, voxel_counts = dask_chunk.chunk_cc3d(nearest_labels, connectivity, False)

    return final, voxel_counts


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
    unique_idx, split_idx = np.unique(chunk_idx, return_index=True, axis=0)
    # remove 0
    split_idx = split_idx[1:]
    splitted = np.split(merged, split_idx, axis=0)
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
        PC_DOWNSAMPLE_RADIUS,
        ANISOTROPY,
        PC_PRED_THRESHOLD,
        CHUNK_SIZE,
        CONNECTIVITY,
    )
    __import__("pdb").set_trace()
