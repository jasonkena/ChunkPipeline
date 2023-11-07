import math
import numpy as np

import kimimaro
from skimage.measure import block_reduce
import sys

sys.path.append("../..")
from skeleton import skel as skel_lib
import scipy
from cloudvolume import Skeleton
import chunk_pipeline.tasks.chunk as chunk

import dask


def _chunk_downsample(vol, downsample_chunk_size, block_info):
    location = block_info[0]["array-location"]
    offset = np.array([x[0] for x in location])
    return [
        block_reduce(vol, block_size=downsample_chunk_size, func=np.max),
        np.array(offset),
    ]


def chunk_downsample(vol, anisotropy, downsample_radius):
    downsample_chunk_size = tuple(
        [math.ceil(downsample_radius / anisotropy[i]) for i in range(3)]
    )
    real_anisotropy = [downsample_chunk_size[i] * anisotropy[i] for i in range(3)]

    downsampled, offsets = chunk.chunk(
        _chunk_downsample,
        [vol],
        output_dataset_dtypes=[object, object],
        downsample_chunk_size=downsample_chunk_size,
    )
    return downsampled, offsets, real_anisotropy


def _chunk_kimimaro(vol, offset, kimi_params, real_anisotropy, anisotropy):
    vol = vol.item()
    offset = offset.item()

    skels = kimimaro.skeletonize(vol, anisotropy=real_anisotropy, **kimi_params)
    # original coordinate system
    for i in skels:
        skels[i].vertices /= np.array(anisotropy)
        skels[i].vertices += offset
    return [skels]


def chunk_kimimaro(vol, offsets, kimi_params, real_anisotropy, anisotropy):
    return chunk.chunk(
        _chunk_kimimaro,
        [vol, offsets],
        output_dataset_dtypes=[object],
        kimi_params=kimi_params,
        real_anisotropy=real_anisotropy,
        anisotropy=anisotropy,
    )


@dask.delayed
def _fast_join_close_components_kernel(s1, s2):
    dist_matrix = scipy.spatial.distance.cdist(s1.vertices, s2.vertices)
    radii = np.min(dist_matrix)
    index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    return radii, index


# adapted from https://github.com/seung-lab/kimimaro/blob/74780a40008a56af3b55ecd8c4f3e4dcbdec6e85/kimimaro/postprocess.py#L79
def fast_join_close_components(skeletons, radius=None):
    """
    Given a set of skeletons which may contain multiple connected components,
    attempt to connect each component to the nearest other component via the
    nearest two vertices. Repeat until no components remain or no points closer
    than `radius` are available.
    radius: float in same units as skeletons
    Returns: Skeleton
    """
    if radius is not None and radius <= 0:
        raise ValueError("radius must be greater than zero: " + str(radius))

    if isinstance(skeletons, Skeleton):
        skeletons = [skeletons]

    skels = []
    for skeleton in skeletons:
        skels += skeleton.components()

    skels = [skl.consolidate() for skl in skels if not skl.empty()]

    if len(skels) == 1:
        return skels[0]
    elif len(skels) == 0:
        return Skeleton()

    while len(skels) > 1:
        N = len(skels)

        radii_matrix = np.zeros((N, N), dtype=np.float32) + np.inf
        index_matrix = np.zeros((N, N, 2), dtype=np.uint32) + -1

        results = {}
        for i in range(N):
            for j in range(i + 1, N):
                results[(i, j)] = _fast_join_close_components_kernel(skels[i], skels[j])
        results = dask.compute(results)[0]

        for i, j in results:
            radii, index = results[(i, j)]
            radii_matrix[i, j] = radii
            radii_matrix[j, i] = radii

            index_matrix[i, j] = index
            index_matrix[j, i] = index

        if np.all(radii_matrix) == np.inf:
            break

        min_radius = np.min(radii_matrix)
        if radius is not None and min_radius > radius:
            break

        i, j = np.unravel_index(np.argmin(radii_matrix), radii_matrix.shape)
        s1, s2 = skels[i], skels[j]
        fused = Skeleton.simple_merge([s1, s2])

        fused.edges = np.concatenate(
            [
                fused.edges,
                [[index_matrix[i, j, 0], index_matrix[i, j, 1] + s1.vertices.shape[0]]],
            ]
        )
        skels[i] = None
        skels[j] = None
        skels = [_ for _ in skels if _ is not None] + [fused]

    return Skeleton.simple_merge(skels).consolidate()


def _chunk_connect_skels(all_skels, fuse_radius, pre_merge=False):
    all_skels = all_skels.flatten().tolist()
    skels = []
    for i in all_skels:
        if isinstance(i, dict):
            for j in i.values():
                skels.append(j)
        else:
            skels.append(i)
    if pre_merge:
        skels = Skeleton.simple_merge(skels).consolidate()

    return [fast_join_close_components(skels, radius=fuse_radius)]


def chunk_connect_skels(skels, fuse_radius):
    return chunk.chunk(
        _chunk_connect_skels,
        [skels],
        output_dataset_dtypes=[object],
        pad="half_extend",
        fuse_radius=fuse_radius,
    )


@dask.delayed
def _aggregate_skels(all_skels, dust_threshold, tick_threshold, fuse_radius, row):
    skel = _chunk_connect_skels(all_skels, fuse_radius, pre_merge=True)[0]
    skel = kimimaro.postprocess(
        skel, dust_threshold=dust_threshold, tick_threshold=tick_threshold
    )
    skel.vertices += np.array([row[1], row[3], row[5]]).reshape(1, 3)

    return skel


@dask.delayed
def _longest_path(skel):
    if len(skel.components()) != 1:
        print("Skeleton has multiple components or is empty")
        return None

    seed = skel_lib.find_furthest_pt(skel, 0, single=False)[0]
    longest_path = skel_lib.find_furthest_pt(skel, seed, single=False)[1][0]
    return longest_path


def task_skeletonize(cfg, extracted, fuse_radius=None):
    vol = extracted["raw"]
    row = extracted["row"]
    general = cfg["GENERAL"]
    kimi = cfg["KIMI"]
    post = kimi["POSTPROCESS_PARAMS"]

    downsampled, offsets, real_anisotropy = chunk_downsample(
        vol, general["ANISOTROPY"], kimi["DOWNSAMPLE_RADIUS"]
    )

    skels = chunk_kimimaro(
        downsampled, offsets, kimi["PARAMS"], real_anisotropy, general["ANISOTROPY"]
    )
    # ensure everything is merged
    skels = chunk_connect_skels(skels, fuse_radius=fuse_radius)

    skel = _aggregate_skels(
        skels,
        post["dust_threshold"],
        post["tick_threshold"],
        post["fuse_radius"],
        row,
    )
    longest_path = _longest_path(skel)
    return {"skeleton": skel, "longest_path": longest_path}
