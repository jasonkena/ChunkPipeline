import itertools
import inspect
import math
import string

import numpy as np
import pandas as pd
import cc3d
from imu.io import get_bb_all3d
from unionfind import UnionFind

import dask
import dask.array as da
import dask.dataframe as df
from distributed import get_client, get_worker

from chunk_pipeline.utils import object_array, publish, normalize_dataset

import logging

import gc


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


def partial_func(func, ndim):
    def _postprocess(result):
        result = object_array(result)
        result = result.reshape(*[1 for _ in range(ndim)], -1)

        return result

    def _inner_partial(*args, **kwargs):
        return _postprocess(func(*args, **kwargs))

    def _inner_partial_block(*args, block_info=None, **kwargs):
        return _postprocess(func(*args, **kwargs, block_info=block_info))

    if "block_info" in [x.name for x in inspect.signature(func).parameters.values()]:
        return _inner_partial_block
    else:
        return _inner_partial


def chunk(
    func,
    input_datasets,
    output_dataset_dtypes=[],
    pad=False,
    pad_width=(1, 1, 1),
    trim_output=True,
    align_idx=None,
    name=None,
    **kwargs,
):
    # func
    """
    input:
        *vols, block_info=None, **kwargs
        -- or
        *vols, **kwargs
    output:
        [*vols, statistic]: if statistic exists
    """
    # input_datasets, list of Dask arrays
    # output_dataset_dtypes, dtype for each output dataset, object for statistics
    # pad: "extend", "half_extend" or False value, output will be trimmed
    # trim_output: whether to undo pad for non-object dtypes, only relevant when pad is True
    # align_arrays: idx for input_datasets that should be aligned

    # NOTE: assumes input sizes are all equal and that output size = input size
    # NOTE: if output_dataset_dtypes is not empty, func must return same shape as input vol
    # NOTE: func should not overwrite input; just pass same dataset as output

    assert pad in ["extend", "half_extend", False]
    if name is None:
        name = func.__name__
    # assert len(set([i.shape for i in input_datasets])) == 1
    if len(output_dataset_dtypes):
        assert len(input_datasets) > 0
    # because trimming does not work reliably for half_extend
    if pad == "half_extend" and not trim_output:
        assert all([i == object for i in output_dataset_dtypes])

    # unify chunks of the different arrays
    if align_idx:
        input_datasets = input_datasets.copy()
        aligned_arrays = [input_datasets[i] for i in align_idx]
        assert len(set([i.shape for i in aligned_arrays])) == 1
        axes = string.ascii_lowercase[: len(aligned_arrays[0].shape)]
        sequence = [(i, axes) for i in aligned_arrays]
        # flatten sequence
        sequence = itertools.chain.from_iterable(sequence)
        _, aligned_arrays = da.core.unify_chunks(*sequence)
        for i in align_idx:
            input_datasets[i] = aligned_arrays[i]

    shape = input_datasets[0].shape
    old_chunks = input_datasets[0].chunks

    if pad:
        if pad == "extend":
            depth = {i: (pad_width[i], pad_width[i]) for i in range(3)}
        elif pad == "half_extend":
            depth = {i: (0, pad_width[i]) for i in range(3)}
        else:
            raise ValueError("pad must be 'extend' or 'half_extend' or False")
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

    kwargs = {
        "dtype": object,
        "chunks": [*(1 for _ in range(len(shape))), len(output_dataset_dtypes)],
        "meta": np.empty(1, dtype=object),
        "new_axis": len(shape),
        "name": f"chunk_output_{name}",
        **kwargs,
    }
    # if overwrite_kwargs:
    #     kwargs = overwrite_kwargs(kwargs)
    # [z, y, x, num_outputs]
    output = da.map_blocks(
        partial_func(func, input_datasets[0].ndim), *input_datasets, **kwargs
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
            name=f"chunk_idx_{name}",
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
def _bbox_aggregate(bboxes, uint_dtype):
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
    return np.array(result, dtype=uint_dtype)


def chunk_bbox(vol, uint_dtype):
    # calculate bbox in chunks
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # returns bboxes in form [seg id, zmin, zmax, etc]

    # restore original coordinates
    # [z,y,x], then [seg id, zmin, zmax, etc]
    bboxes = chunk(_chunk_bbox, [vol], output_dataset_dtypes=[object])
    bboxes = da.from_delayed(
        _bbox_aggregate(bboxes, uint_dtype), shape=(np.nan, 7), dtype=uint_dtype
    )
    return bboxes


def _chunk_cc3d_neighbors(partial_cc3d, partial_statistics, neighbors, block_info):
    chunk_idx = block_info[0]["chunk-location"]
    num_chunks = block_info[0]["num-chunks"]

    # TODO: refactor mask generation into separate function
    mask = np.zeros(partial_cc3d.shape, dtype=bool)
    # increment zyx_idx if not on boundary
    if chunk_idx[0] != num_chunks[0] - 1:
        mask[-2:, :, :] = True
    if chunk_idx[1] != num_chunks[1] - 1:
        mask[:, -2:, :] = True
    if chunk_idx[2] != num_chunks[2] - 1:
        mask[:, :, -2:] = True

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
        uf_union = np.unique(np.concatenate(pairs, axis=0), axis=0)
    else:
        uf_union = np.array((), dtype=neighbors.dtype).reshape((0, 4, 2))

    # do not return remapping, it can be constructed
    return [uf_add, uf_union]


def chunk_remap(vol, remapping):
    return chunk(
        lambda vol, remapping: [
            remapping.item()[vol] if remapping.dtype == object else remapping[vol]
        ],
        [vol, remapping],
        [vol.dtype],
        name="remap",
    )


def _chunk_half_extend_cc3d(vol, connectivity, uint_dtype, block_info):
    chunk_idx = block_info[0]["chunk-location"]
    num_chunks = block_info[0]["num-chunks"]

    zyx_idx_mask = np.zeros(list(vol.shape) + [3], dtype=int)
    mask = np.zeros(vol.shape, dtype=bool)
    # increment zyx_idx if not on boundary
    if chunk_idx[0] != num_chunks[0] - 1:
        zyx_idx_mask[-1, :, :, 0] = 1
        mask[-2:, :, :] = True
    if chunk_idx[1] != num_chunks[1] - 1:
        zyx_idx_mask[:, -1, :, 1] = 1
        mask[:, -2:, :] = True
    if chunk_idx[2] != num_chunks[2] - 1:
        zyx_idx_mask[:, :, -1, 2] = 1
        mask[:, :, -2:] = True
    zyx_idx_mask = zyx_idx_mask + np.array(block_info[0]["chunk-location"]).reshape(
        1, 1, 1, 3
    )

    connected_components = cc3d.connected_components(vol, connectivity=connectivity)

    stacked = np.concatenate(
        [zyx_idx_mask, np.expand_dims(connected_components, -1)], axis=-1
    )
    neighbors = stacked[mask]

    return [connected_components.astype(uint_dtype), neighbors.astype(uint_dtype)]


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

        is_valid = idx[:, 3] < index_ragged(voxel_statistics, lambda x: x.shape[0])
        if not np.all(is_valid):
            logging.warning("IndexError in voxel_counts")
        new_counts = index_ragged(voxel_statistics[is_valid], idx[is_valid][:, 3])
        voxel_counts[i] += np.sum(new_counts)

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


def chunk_cc3d(vol, connectivity, k, uint_dtype):
    # perform chunked cc3d calculations
    # vol: dataset used as input
    # k: either int or False
    # NOTE: if IndexError is emitted, result is invalid

    chunk_size = vol.chunksize
    # cc3d only handles inputs > 1
    assert all([i > 1 for i in chunk_size])

    partial_cc3d, neighbors = chunk(
        _chunk_half_extend_cc3d,
        [vol],
        [uint_dtype, object],
        pad="half_extend",
        connectivity=connectivity,
        uint_dtype=uint_dtype,
    )

    partial_statistics = chunk(
        lambda vol: [cc3d.statistics(vol.astype(uint_dtype))],
        [partial_cc3d],
        [object],
        name="partial_statistics",
    )

    uf_add, uf_union = chunk(
        _chunk_cc3d_neighbors,
        [partial_cc3d, partial_statistics, neighbors],
        [object, object],
        pad="half_extend",
    )

    temp = compute_remapping(uf_add, uf_union, partial_statistics, vol.shape, k)
    remapping, voxel_counts = temp[0], temp[1]
    remapping = da.from_delayed(remapping, shape=list(vol.numblocks), dtype=object)
    remapping = da.rechunk(remapping, chunks=(1, 1, 1))

    partial_cc3d = chunk_remap(partial_cc3d, remapping)
    voxel_counts = da.from_delayed(voxel_counts, shape=[np.nan], dtype=int)

    return partial_cc3d, voxel_counts


# def _chunk_nonzero(vol, uint_dtype, extra=None, block_info=None):
#     # kwargs must contain func, and it must return 2 things: binary mask and another (array or None)
#     # returns indices where vol is true
#     # [3, N]
#     idx = np.stack(np.nonzero(vol), axis=1)
#     if extra is not None:
#         extra = extra[vol.astype(bool)]
#     idx = idx + np.array(
#         [block_info[0]["array-location"][i][0] for i in range(3)]
#     ).reshape(-1, 3)
#     if extra is not None:
#         idx = np.concatenate([idx, extra.reshape(-1, 1)], axis=1)
#
#     return [idx.astype(uint_dtype)]
#
#
# @dask.delayed
# def _aggregate_nonzero(idx):
#     idx = np.concatenate(idx.flatten(), axis=0)
#     # for ordering purposes
#     return np.unique(idx, axis=0)
#
#
# def chunk_nonzero(vol, uint_dtype, extra=None):
#     # TODO: implement chunked saving instead of aggregating all indices
#     inputs = [vol, extra] if extra is not None else [vol]
#     result = chunk(
#         _chunk_nonzero,
#         inputs,
#         [object],
#         align_idx=([0, 1] if extra is not None else None),
#         uint_dtype=uint_dtype,
#     )
#     result = _aggregate_nonzero(result)
#     result = da.from_delayed(
#         result, shape=[np.nan, 3 + (extra is not None)], dtype=uint_dtype
#     )


def get_seg(vol, bbox, chunk_size, filter_id):
    if filter_id:
        vol = vol == bbox[0]
    # extra +1 due to bbox format
    result = vol[bbox[1] : bbox[2] + 1, bbox[3] : bbox[4] + 1, bbox[5] : bbox[6] + 1]
    return result.rechunk(chunk_size)


def merge_seg(output, vol, bbox, merge_func):
    slices = tuple(slice(bbox[2 * i + 1], bbox[2 * i + 2] + 1) for i in range(3))
    output[slices] = merge_func(output[slices], vol)
    return output


def _chunk_unique(vol, return_inverse):
    output = np.unique(vol, return_inverse=return_inverse)

    if not return_inverse:
        return [output]

    return [
        output[0],
        output[1].reshape(vol.shape),
    ]


@dask.delayed
def _aggregate_unique(unique):
    return np.unique(np.concatenate(unique.flatten()))


def _chunk_unique_remap(unique, final_unique):
    unique = unique.item()
    return [np.searchsorted(final_unique, unique)]


def chunk_unique(vol, return_inverse, uint_dtype):
    # since dask's unique is expensive
    # simple scalar unique (no axis parameter in unique)
    # return_inverse: True or False
    # NOTE: will need to rewrite if number of unique values exceed memory
    output = chunk(
        _chunk_unique,
        [vol],
        output_dataset_dtypes=[object, uint_dtype] if return_inverse else [object],
        return_inverse=return_inverse,
    )

    if not return_inverse:
        unique = output
    else:
        unique, idx = output

    final_unique = da.from_delayed(
        _aggregate_unique(unique), shape=[np.nan], dtype=unique.dtype
    )

    if not return_inverse:
        return final_unique

    remapping = chunk(
        _chunk_unique_remap,
        [unique],
        output_dataset_dtypes=[object],
        final_unique=final_unique,
    )

    inverse = chunk_remap(idx, remapping)

    return final_unique, inverse


def _chunk_max_pool(vol):
    return [np.max(vol)]


def chunk_downsample(vol, chunk_width, anisotropy):
    # chunk_width is in nanometers
    chunk_size = [math.ceil(chunk_width / anisotropy[i]) for i in range(3)]
    vol = da.rechunk(vol, chunks=tuple(chunk_size))

    downsampled = chunk(_chunk_max_pool, [vol], [object])
    # an approximation of chunk_width
    real_anisotropy = [chunk_size[i] * anisotropy[i] for i in range(3)]
    return downsampled, real_anisotropy


def _chunk_seed_groupby(dataframe, chunk_size, dtype):
    client = get_client()

    # breaks if np.zeros is used instead
    # raises "TypeError('string indices must be integers')"
    vol = np.zeros(math.prod(chunk_size), dtype=dtype)
    z, y, x = [dataframe[v] for v in ["rel_z", "rel_y", "rel_x"]]
    z, y, x = [np.array(v).astype(int) for v in [z, y, x]]
    idx = z * chunk_size[1] * chunk_size[2] + y * chunk_size[2] + x
    vol[idx] = np.array(dataframe["value"])
    vol = vol.reshape(chunk_size)
    # vol = da.from_array(vol, chunks=chunk_size)

    name = publish("chunk_seed_groupby", vol)

    return pd.DataFrame({"obj": [name]}, index=[0])


@dask.delayed
def _chunk_seed_write(shape, chunk_size, dataframe, dtype):
    client = get_client()
    blank = da.zeros(shape, dtype=dtype, chunks=chunk_size).blocks
    blocks = []
    for i in range(blank.shape[0]):
        i_blocks = []
        for j in range(blank.shape[1]):
            j_blocks = []
            for k in range(blank.shape[2]):
                j_blocks.append(blank[i, j, k])
            i_blocks.append(j_blocks)
        blocks.append(i_blocks)

    objs = list(dataframe["obj"])
    for i, (z, y, x, _) in enumerate(dataframe.index):
        blocks[z][y][x] = client.get_dataset(objs[i])[
            : blocks[z][y][x].shape[0],
            : blocks[z][y][x].shape[1],
            : blocks[z][y][x].shape[2],
        ]
        client.unpublish_dataset(objs[i])

    result = da.block(blocks)

    name = publish("new_chunk_seed_write", result, persist=True)
    return name


# @dask.delayed
# def _chunk_seed_write(shape, chunk_size, dataframe, dtype):
#     client = get_client()
#
#     vol = da.zeros(shape, chunks=chunk_size, dtype=dtype)
#     objs = list(dataframe["obj"])
#     for i, (z, y, x, _) in enumerate(dataframe.index):
#         z0 = z * chunk_size[0]
#         y0 = y * chunk_size[1]
#         x0 = x * chunk_size[2]
#         z1 = min(z0 + chunk_size[0], shape[0])
#         y1 = min(y0 + chunk_size[1], shape[1])
#         x1 = min(x0 + chunk_size[2], shape[2])
#
#         dataset = client.get_dataset(objs[i])
#         logging.error(f"write dataset: {dataset}")
#
#         vol[z0:z1, y0:y1, x0:x1] = dataset[: z1 - z0, : y1 - y0, : x1 - x0]
#
#     name = publish("chunk_seed_write", vol, persist=True)
#     # dask.compute(vol)
#     return name
#


def groupby_chunk_seed(shape, points, values, chunk_size, dtype):
    # shape: (z, y, x)
    # points: (np.nan, 3)
    # values: (np.nan)
    # dtype is not necessarily uint
    chunk_idx = da.floor(points / np.array(chunk_size).reshape(1, -1)).astype(int)
    rel_points = points - chunk_idx * np.array(chunk_size).reshape(1, -1)

    # merged = da.concatenate(
    #     [chunk_idx, rel_points, values[:, np.newaxis]],
    #     axis=1,
    #     allow_unknown_chunksizes=True,
    # )
    col_names = [
        ["chunk_z", "chunk_y", "chunk_x"],
        ["rel_z", "rel_y", "rel_x"],
        ["value"],
    ]
    columns = [
        df.from_dask_array(x, columns=col_names[i])
        for i, x in enumerate([chunk_idx, rel_points, values[:, np.newaxis]])
    ]
    dataframe = df.concat(columns, axis=1, ignore_unknown_divisions=True)

    dataframe = dataframe.groupby(["chunk_z", "chunk_y", "chunk_x"]).apply(
        _chunk_seed_groupby, meta={"obj": object}, chunk_size=chunk_size, dtype=dtype
    )
    name = _chunk_seed_write(shape, chunk_size, dataframe, dtype)
    result = normalize_dataset(name, shape, dtype, chunk_size)

    return result


def chunk_seed(shape, points, values, chunk_size, dtype):
    # does not work well, since flattening ruins chunking
    if np.isnan(points.shape).any():
        points.compute_chunk_sizes()
    if np.isnan(values.shape).any():
        values.compute_chunk_sizes()

    # round shape to nearest multiple of chunk_size
    round_shape = [
        math.ceil(shape[i] / chunk_size[i]) * chunk_size[i] for i in range(3)
    ]

    # total number of chunks
    tz, ty, tx = [round_shape[i] // chunk_size[i] for i in range(3)]
    # chunk_idx
    nz, ny, nx = [points[:, i] // chunk_size[i] for i in range(3)]
    # remainder
    z, y, x = [points[:, i] % chunk_size[i] for i in range(3)]
    cz, cy, cx = [chunk_size[i] for i in range(3)]

    # index as sum of starting chunk index and remainder
    idx = (nz * ty * tx + ny * tx + nx) * (cz * cy * cx) + (z * cy * cx) + (y * cx) + x

    # NOTE: use this to seed pre-existing volume
    # def _chunk_reshape(vol, shape):
    #     return [vol.reshape(shape)]

    # vol = da.zeros(round_shape, chunks=chunk_size, dtype=dtype)
    # # [tz, ty, tz] -> [1]
    # flattened = chunk(_chunk_reshape, [vol], output_dataset_dtypes=[object]).reshape(-1)
    # flattened = da.map_blocks(lambda x: x, flattened, chunks=(cz*cy*cx,), dtype=dtype)

    flattened = da.zeros(
        tz * ty * tx * cz * cy * cx, chunks=(cz * cy * cx,), dtype=dtype
    )
    # https://github.com/dask/dask/pull/3407
    flattened[(idx,)] = values
    print(flattened)
    flattened = flattened.reshape([tz, ty, tx, cz, cy, cx])
    print(flattened)

    # [tz, cz, ty, cy, tx, cx]
    flattened = flattened.transpose([0, 3, 1, 4, 2, 5])

    vol = flattened.reshape([tz * cz, ty * cy, tx * cx])
    vol = vol[: shape[0], : shape[1], : shape[2]]  # undo padding

    print(vol)
    return vol


# def naive_chunk_seed(shape, points, values, chunk_size, dtype):
#     shape, points = dask.compute(shape, points)
#     z, y, x = [points[:, i].tolist() for i in range(3)]
#     vol = da.zeros(shape, dtype=dtype, chunks=chunk_size)
#     vol.vindex[z, y, x] = values
#
#     return vol
