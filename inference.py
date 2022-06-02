import numpy as np
import h5py
from chunk_sphere import get_dt
from utils import create_compressed, extend_bbox
from settings import *
import os
import chunk
import chunk_sphere
import math


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


def _chunk_seed(params, vol, points, pred):
    chunk_size = params["chunk_size"]
    z_min, y_min, x_min = [
        params[i] * chunk_size[idx] for idx, i in enumerate(["z", "y", "x"])
    ]
    z_max, y_max, x_max = (
        z_min + chunk_size[0],
        y_min + chunk_size[1],
        x_min + chunk_size[2],
    )
    lower_bound = np.array([z_min, y_min, x_min]).reshape(1, -1)
    upper_bound = np.array([z_max, y_max, x_max]).reshape(1, -1)

    # [N, 3]
    in_range = np.logical_and(
        points >= lower_bound,
        points < upper_bound,
    )
    # [N]
    in_range = np.all(in_range, axis=1)
    valid_points = points[in_range] - lower_bound
    valid_pred = pred[in_range]

    result = np.zeros_like(vol)
    result[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]] = valid_pred

    return [result]


def _chunk_max_pool(params, vol, output):
    z, y, x = [params[i] for i in ["z", "y", "x"]]
    output[z, y, x] = np.max(vol)
    return []


def max_dt(vol, chunk_width, anisotropy, num_workers):
    # chunk_width is in nanometers
    chunk_size = [math.ceil(chunk_width / anisotropy[i]) for i in range(3)]
    output_shape = [math.ceil(vol.shape[i] / chunk_size[i]) for i in range(3)]
    output = np.zeros(output_shape)

    chunk.simple_chunk(
        [],
        [vol],
        chunk_size,
        _chunk_max_pool,
        num_workers,
        pass_params=True,
        output=output,
    )

    # an approximation of chunk_width
    real_anisotropy = [chunk_size[i] * anisotropy[i] for i in range(3)]
    dt = chunk_sphere._get_dt(
        output, real_anisotropy, black_border=False, filter_idx=None
    )
    return np.max(dt)


def _chunk_nearest(vol, trunk_dt, spine_dt):
    # background 0, trunk 1, spine 2
    return [((spine_dt > trunk_dt) + 1) * vol]


def inference(
    base_path,
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
    output = h5py.File(os.path.join(base_path, "point_pred.h5"), "w")
    shape = vol_dataset.shape
    seeded = create_compressed(
        output,
        "seeded",
        shape=shape,
        dtype="uint16",
    )
    # TODO: TODO: TODO: implement smarter in range points, use a dictionary like you did for uinion find remapping
    seeded = chunk.simple_chunk(
        [seeded],
        [seeded],
        chunk_size,
        _chunk_seed,
        num_workers,
        pass_params=True,
        points=points,
        pred=pred,
    )

    threshold = max_dt(vol_dataset, downsample_radius, anisotropy, num_workers)
    trunk_dt = get_dt(
        # NOTE: NOTE: NOTE: should not be uint16
        create_compressed(output, "trunk_dt", shape=shape, dtype="f"),
        seeded,
        chunk_size,
        anisotropy,
        False,
        threshold,
        num_workers,
        filter_idx=1,
    )
    spine_dt = get_dt(
        create_compressed(output, "spine_dt", shape=shape, dtype="f"),
        seeded,
        chunk_size,
        anisotropy,
        False,
        threshold,
        num_workers,
        filter_idx=2,
    )

    nearest_labels = chunk.simple_chunk(
        [create_compressed(output, "nearest_labels", shape=shape, dtype="uint16")],
        [vol_dataset, trunk_dt, spine_dt],
        chunk_size,
        _chunk_nearest,
        num_workers,
    )

    final, voxel_counts = chunk.chunk_cc3d(
        create_compressed(output, "main", shape=shape, dtype="uint16"),
        nearest_labels,
        output,
        chunk_size,
        connectivity,
        num_workers,
        k=False,
    )

    return final


def dumb_predict(points):
    return np.random.rand(*points.shape[:2])
    # takes numpy array [N, num_points, 3]


if __name__ == "__main__":
    h5 = h5py.File("dumb/1.h5", "r")
    vol_dataset = h5.get("main")
    row = h5.get("row")
    points = np.load("dumb/1.npy")[:30000]
    final = inference(
        "dumb",
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
    __import__("pdb").set_trace()
