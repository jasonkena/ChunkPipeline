import numpy as np
from utils import pad_vol
import dask
import dask.array as da
import itertools
from imu.io import get_bb_all3d
import cc3d
from unionfind import UnionFind

import math


def object_array(input):
    # properly create an object array
    result = np.empty(len(input), dtype=object)
    result[:] = input
    return result


def index_ragged(vol, idx, object_dtype=False):
    # assuming 1d input
    if type(idx) is np.ndarray:
        result = [x[idx[i]] for i, x in enumerate(vol)]
    elif callable(idx):
        result = [idx(x) for x in vol]
    else:
        result = [x[idx] for x in vol]
    if object_dtype:
        return object_array(result)
    return np.array(result)


def partial_func(func):
    def inner_partial_func(*args, block_info=None, **kwargs):
        temp = func(*args, **kwargs, block_info=block_info)
        result = object_array(temp)
        result = result.reshape(
            *[1 for _ in range(len(block_info[0]["num-chunks"]))], -1
        )

        return result

    return inner_partial_func


def chunk(
    func,
    input_datasets,
    output_dataset_dtypes=[],
    pad=False,
    pad_width=(1, 1, 1),
    trim_output=True,
    **kwargs
):
    # func
    """
    input:
        *vols, block_info=None, **kwargs
    output:
        [*vols, statistic]: if statistic exists
    """
    # input_datasets, list of Dask arrays
    # output_dataset_dtypes, dtype for each output dataset, object for statistics
    # pad: "extend", "half_extend" or False value, output will be trimmed
    # trim_output: whether to undo pad for non-object dtypes, only relevant when pad is True

    # NOTE: assumes input sizes are all equal and that output size = input size
    # NOTE: if output_dataset_dtypes is not empty, func must return same shape as input vol
    # NOTE: func should not overwrite input; just pass same dataset as output

    assert pad in ["extend", "half_extend", False]
    # assert len(set([i.shape for i in input_datasets])) == 1
    if len(output_dataset_dtypes):
        assert len(input_datasets) > 0
    # because trimming does not work reliably for half_extend
    if pad == "half_extend" and not trim_output:
        assert all([i == object for i in output_dataset_dtypes])
    shape = input_datasets[0].shape
    old_chunks = input_datasets[0].chunks

    if pad:
        if pad == "extend":
            depth = {i: (pad_width[i], pad_width[i]) for i in range(3)}
        elif pad == "half_extend":
            depth = {i: (0, pad_width[i]) for i in range(3)}
        # do not overlap object datasets
        input_datasets = [
            (
                da.overlap.overlap(x, depth=depth, boundary="none")
                if x.dtype != object
                else x
            )
            for x in input_datasets
        ]
    new_chunks = input_datasets[0].chunks

    # [z, y, x, num_outputs]
    output = da.map_blocks(
        partial_func(func),
        *input_datasets,
        dtype=object,
        chunks=[*(1 for _ in range(len(shape))), len(output_dataset_dtypes)],
        meta=np.empty(1, dtype=object),
        new_axis=len(shape),
        **kwargs
    )

    final = []
    for idx, dtype in enumerate(output_dataset_dtypes):
        if dtype == object:
            chunks = [1 for _ in range(len(shape))]
        else:
            chunks = new_chunks
            if pad and not trim_output:
                chunks = old_chunks

        result = da.map_blocks(
            lambda x, idx, ddtype: x[..., idx].item()
            if ddtype != object
            else x[..., idx],
            output,
            dtype=dtype,
            chunks=chunks,
            meta=np.empty(1, dtype=dtype),
            drop_axis=len(shape),
            idx=idx,
            ddtype=dtype,
        )

        if dtype != object:
            if pad and trim_output:
                result = da.overlap.trim_overlap(result, depth=depth, boundary="none")
        final.append(result)

    if len(final) == 1:
        final = final[0]
    return final


def get_is_first_unique(array):
    # assuming sorted, return whether an element is the first unique element
    assert np.all(array != 0)
    return np.pad(array[:-1], (1, 0)) != array


def _chunk_bbox(vol, block_info):
    # add offset to bounding boxes based on zyx
    bboxes = get_bb_all3d(vol)
    bboxes[:, 1:3] += block_info[0]["array-location"][0][0]
    bboxes[:, 3:5] += block_info[0]["array-location"][1][0]
    bboxes[:, 5:7] += block_info[0]["array-location"][2][0]

    return [bboxes]


@dask.delayed
def _bbox_aggregate(bboxes):
    bboxes = np.concatenate(bboxes.reshape(-1).tolist(), axis=0)
    bboxes = bboxes[np.argsort(bboxes[:, 0])]
    assert not np.any(bboxes[:, 0] == 0)

    is_first_unique = get_is_first_unique(bboxes[:, 0])
    idx = np.nonzero(is_first_unique)[0].tolist() + [is_first_unique.shape[0]]
    result = []

    for i in range(len(idx) - 1):
        result.append(bboxes[idx[i]])
        for j in [1, 3, 5]:
            result[-1][j] = np.min(bboxes[idx[i] : idx[i + 1], j])
        for j in [2, 4, 6]:
            result[-1][j] = np.max(bboxes[idx[i] : idx[i + 1], j])
    return np.array(result, dtype=int)


def chunk_bbox(vol):
    # calculate bbox in chunks
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # returns bboxes in form [seg id, zmin, zmax, etc]

    # restore original coordinates
    # [z,y,x], then [seg id, zmin, zmax, etc]
    bboxes = chunk(_chunk_bbox, [vol], output_dataset_dtypes=[object])
    bboxes = da.from_delayed(_bbox_aggregate(bboxes), shape=(np.nan, 7), dtype=int)
    return bboxes


def _chunk_cc3d_neighbors(
    partial_cc3d, partial_statistics, neighbors, mask, block_info
):
    mask = mask[
        : partial_cc3d.shape[0], : partial_cc3d.shape[1], : partial_cc3d.shape[2]
    ]

    partial_statistics = partial_statistics.item()
    neighbors = neighbors.item()

    remapping = np.arange(
        partial_statistics["voxel_counts"].shape[0], dtype=partial_cc3d.dtype
    )
    uf_add = np.concatenate(
        [
            np.repeat(
                np.array(
                    block_info[0]["chunk-location"], dtype=partial_cc3d.dtype
                ).reshape(1, -1),
                remapping.shape[0],
                axis=0,
            ),
            remapping.reshape(-1, 1),
        ],
        axis=1,
    )[
        1:
    ]  # remove 0 component

    zyx_idx = neighbors[:, :-1]
    new_cc_idx = partial_cc3d[mask]
    old_cc_idx = neighbors[:, -1]

    order = np.argsort(old_cc_idx)
    zyx_idx, new_cc_idx, old_cc_idx = [
        i[order] for i in [zyx_idx, new_cc_idx, old_cc_idx]
    ]

    is_valid = new_cc_idx != 0
    zyx_idx, new_cc_idx, old_cc_idx = [
        i[is_valid] for i in [zyx_idx, new_cc_idx, old_cc_idx]
    ]

    result = np.concatenate((zyx_idx, new_cc_idx.reshape(-1, 1)), axis=-1).reshape(
        -1, zyx_idx.shape[-1] + 1
    )

    # union pairs which are have the same old_cc_idx
    unique_idx = get_is_first_unique(old_cc_idx)
    splitted = np.split(result, np.nonzero(unique_idx)[0], axis=0)
    # [num_splits, num_neighbors, idx]
    splitted = [i for i in splitted if i.shape[0] > 1]
    # [num_splits, num_neighbors, idx, 2]
    pairs = [np.stack((i[:-1], i[1:]), axis=-1) for i in splitted]
    # [num_pairs, idx, 2]
    if len(pairs):
        uf_union = np.concatenate(pairs, axis=0)
    else:
        uf_union = np.array((), dtype=neighbors.dtype).reshape((0, 4, 2))

    # do not return remapping, it can be constructed
    return [uf_add, uf_union]


def chunk_remap(vol, remapping):
    return chunk(
        lambda vol, remapping, block_info: [remapping.item()[vol]],
        [vol, remapping],
        [remapping.dtype],
    )


def _chunk_half_extend_cc3d(vol, zyx_idx_mask, mask, connectivity, block_info):
    connected_components = cc3d.connected_components(vol, connectivity=connectivity)
    # trim mask to fit chunk
    mask = mask[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    zyx_idx_mask = zyx_idx_mask + np.array(block_info[0]["chunk-location"]).reshape(
        1, 1, 1, 3
    )
    zyx_idx_mask = zyx_idx_mask[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    neighbors = np.concatenate(
        (
            zyx_idx_mask[mask],
            connected_components[mask].reshape(-1, 1),
        ),
        axis=-1,
    ).reshape(-1, zyx_idx_mask.shape[-1] + 1)

    return [connected_components, neighbors]


@dask.delayed
def compute_remapping(uf_add, uf_union, partial_statistics, vol_shape, k):
    uf_add = np.concatenate(uf_add.flatten().tolist(), axis=0)
    uf_union = np.concatenate(uf_union.flatten().tolist(), axis=0)
    # remove duplicates
    uf_union = np.unique(uf_union, axis=0)

    # disjoint sets to merge components
    uf = UnionFind()
    for i in uf_add:
        uf.add(tuple(i))
    for i in uf_union:
        uf.union(tuple(i[:, 0]), tuple(i[:, 1]))

    # calculate voxel_counts of merged components
    # list of sets containing components
    mapping = np.array(uf.components(), dtype=object)
    # include 0 in voxel_counts
    voxel_counts = np.zeros(mapping.shape[0], dtype=int)
    for i in range(mapping.shape[0]):
        idx = np.array(list(mapping[i]))
        voxel_statistics = partial_statistics[idx[:, 0], idx[:, 1], idx[:, 2]]
        voxel_statistics = index_ragged(
            voxel_statistics, "voxel_counts", object_dtype=True
        )
        new_counts = index_ragged(voxel_statistics, idx[:, 3])

        is_valid = idx[:, 3] < index_ragged(voxel_statistics, lambda x: x.shape[0])
        if not np.all(is_valid):
            print("exists invalid unionfind, not a problem")
        voxel_counts[i] += np.sum(new_counts[is_valid])

    order = np.argsort(voxel_counts)[::-1]
    voxel_counts = voxel_counts[order]
    mapping = mapping[order]

    voxel_counts = np.concatenate(
        (
            [vol_shape[0] * vol_shape[1] * vol_shape[2] - np.sum(voxel_counts)],
            voxel_counts,
        )
    )

    voxel_counts = voxel_counts[: np.max(np.nonzero(voxel_counts)[0]) + 1]
    assert np.all(voxel_counts[1:] > 0)

    # columns [z, y, x, old_idx, new_idx]
    # +1 because of background
    mapping = np.concatenate(
        [
            np.pad(np.array(list(x)), ((0, 0), (0, 1)), constant_values=i + 1)
            for i, x in enumerate(mapping)
        ],
        axis=0,
    )
    # number of old components
    num_components = np.max(mapping[:, -2])
    # +1 because of background
    remapping = np.zeros(
        list(partial_statistics.shape) + [num_components + 1], dtype=mapping.dtype
    )
    remapping[:] = np.arange(num_components + 1)

    remapping[mapping[:, 0], mapping[:, 1], mapping[:, 2], mapping[:, 3]] = mapping[
        :, 4
    ]
    if k:
        remapping = remapping * (remapping <= k)

    # create object_array to chunk
    shape = remapping.shape
    remapping = remapping.reshape(-1, shape[-1])
    remapping = object_array(list(remapping))
    remapping = remapping.reshape(shape[:-1])

    if k:
        voxel_counts[0] += np.sum(voxel_counts[k + 1 :])
        voxel_counts = voxel_counts[: k + 1]

    return remapping, voxel_counts


def chunk_cc3d(vol, connectivity, k):
    # perform chunked cc3d calculations
    # vol: dataset used as input
    # k: either int or False
    # NOTE: if IndexError is emitted, result is invalid

    chunk_size = vol.chunksize
    # cc3d only handles inputs > 1
    assert all([i > 1 for i in chunk_size])

    zyx_idx_mask = np.zeros([i + 1 for i in chunk_size] + [3], dtype=int)
    zyx_idx_mask[-1:, :, :, 0] = 1
    zyx_idx_mask[:, -1:, :, 1] = 1
    zyx_idx_mask[:, :, -1:, 2] = 1

    # mask for half extend
    mask = np.ones([i + 1 for i in chunk_size], dtype=bool)
    # 2 instead of 1 because of [[0,1], [1,0]] edge case
    mask[:-2, :-2, :-2] = False

    partial_cc3d, neighbors = chunk(
        _chunk_half_extend_cc3d,
        [vol],
        [int, object],
        pad="half_extend",
        zyx_idx_mask=zyx_idx_mask,
        mask=mask,
        connectivity=connectivity,
    )

    partial_statistics = chunk(
        lambda vol, block_info: [cc3d.statistics(vol.astype(np.uint64))],
        [partial_cc3d],
        [object],
    )

    uf_add, uf_union = chunk(
        _chunk_cc3d_neighbors,
        [partial_cc3d, partial_statistics, neighbors],
        [object, object],
        pad="half_extend",
        mask=mask,
    )

    temp = compute_remapping(uf_add, uf_union, partial_statistics, vol.shape, k)
    remapping, voxel_counts = temp[0], temp[1]
    remapping = da.from_delayed(remapping, shape=list(vol.numblocks), dtype=object)
    remapping = da.rechunk(remapping, chunks=(1, 1, 1))

    partial_cc3d = chunk_remap(partial_cc3d, remapping)

    return partial_cc3d, voxel_counts


def _chunk_nonzero(vol, block_info):
    # kwargs must contain func, and it must return 2 things: binary mask and another (array or None)
    # returns indices where vol is true
    # [3, N]
    idx = np.stack(np.argwhere(vol), axis=0)
    idx = idx + np.array(
        [block_info[0]["array-location"][i][0] for i in range(3)]
    ).reshape(-1, 3)

    return [idx]


@dask.delayed
def _aggregate_nonzero(idx):
    idx = np.concatenate(idx.flatten(), axis=0)
    # for ordering purposes
    return np.unique(idx, axis=0)


def chunk_nonzero(vol):
    # TODO: implement chunked saving instead of aggregating all indices
    result = chunk(_chunk_nonzero, [vol], [object])
    result = _aggregate_nonzero(result)
    result = da.from_delayed(result, shape=[np.nan, 3], dtype=int)
    return result


# skipping chunk_unique


def get_seg(vol, bbox, filter_id):
    if filter_id:
        vol = vol == bbox[0]
    # extra +1 due to bbox format
    return vol[bbox[1] : bbox[2] + 1, bbox[3] : bbox[4] + 1, bbox[5] : bbox[6] + 1]


def merge_seg(output, vol, bbox, merge_func):
    slices = tuple(slice(bbox[2 * i + 1], bbox[2 * i + 2] + 1) for i in range(3))
    output[slices] = merge_func(output[slices], vol)
    return output
