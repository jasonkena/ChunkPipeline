import numpy as np
import h5py
from utils import pad_vol
from imu.io import get_bb_all3d
from unionfind import UnionFind
import itertools
import cc3d

import math


def extend_slices(slices, half):
    # computes slices to extend slice to include 1 neighboring voxel, and corresponding slice to undo it, restoring original size
    # slices: input slices from np.s_
    # half: whether to only extend right side
    # returns new_slices and shrink_slices
    new_slices = tuple(slice(max(0, i.start - (not half)), i.stop + 1) for i in slices)
    is_left_extended = [int((i.start > 0) and (not half)) for i in slices]
    distances = [i.stop - i.start for i in slices]

    shrink_slices = tuple(
        slice(
            is_left_extended[i],
            is_left_extended[i] + distances[i],
        )
        for i in range(len(slices))
    )

    return new_slices, shrink_slices


def simple_chunk(
    dataset_output, dataset_inputs, chunk_size, pad, pass_params, func, *args, **kwargs
):
    # chunks operations (func) on input
    # dataset_output: output dataset to write to; if None, will write to nested array
    # dataset_inputs: list of h5 datasets; if non-list, will be treated as size of input for zyx
    # chunk_size: [size_z, size_y, size_x]
    # pad: "zero", "extend", "half_extend" or False value, output will be trimmed
    # zero will always pad each dimension by 2, extend will do the same if not on boundary, half_extend right_extends
    # func, *args, **kwargs: self-explanatory

    # NOTE: assumes input sizes are all equal and that output size = input size
    # NOTE: func may overwrite inputs, beware if attempting to parallelize

    if dataset_output == None:
        dataset_output = []
    if isinstance(dataset_inputs, list):
        shape = dataset_inputs[0].shape
    else:
        shape = dataset_inputs
    n_iter = [math.ceil(shape[i] / chunk_size[i]) for i in range(3)]

    for z in range(n_iter[0]):
        for y in range(n_iter[1]):
            for x in range(n_iter[2]):
                original_slices = np.s_[
                    z * chunk_size[0] : (z + 1) * chunk_size[0],
                    y * chunk_size[1] : (y + 1) * chunk_size[1],
                    x * chunk_size[2] : (x + 1) * chunk_size[2],
                ]
                if pad == "extend" or pad == "half_extend":
                    slices, shrink_slices = extend_slices(
                        original_slices, pad == "half_extend"
                    )
                else:
                    slices = original_slices

                if isinstance(dataset_inputs, list):
                    inputs = [i[slices] for i in dataset_inputs]
                else:
                    inputs = []
                if pad == "zero":
                    inputs = [pad_vol(i, [3, 3, 3]) for i in inputs]

                if pass_params:
                    output = func(z, y, x, chunk_size, *inputs, *args, **kwargs)
                else:
                    output = func(*inputs, *args, **kwargs)

                if not isinstance(dataset_output, list):
                    if pad == "zero":
                        output = output[1:-1, 1:-1, 1:-1]
                    elif pad == "extend" or pad == "half_extend":
                        output = output[shrink_slices]

                if isinstance(dataset_output, list):
                    dataset_output.append(output)
                else:
                    dataset_output[original_slices] = output

    if isinstance(dataset_output, list):
        dataset_output = np.array(dataset_output, dtype=object)
        return dataset_output.reshape(*n_iter)
    return dataset_output


def get_is_first_unique(array):
    # assuming sorted, return whether an element is the first unique element
    assert np.all(array != 0)
    return np.pad(array[:-1], (1, 0)) != array


def _chunk_bbox(z, y, x, chunk_size, vol):
    # add offset to bounding boxes based on zyx
    bboxes = get_bb_all3d(vol)
    bboxes[:, 1:3] += chunk_size[0] * z
    bboxes[:, 3:5] += chunk_size[1] * y
    bboxes[:, 5:7] += chunk_size[2] * x

    return bboxes


def chunk_bbox(vol, chunk_size):
    # calculate bbox in chunks
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # returns bboxes in form [seg id, zmin, zmax, etc]

    # restore original coordinates
    # [z,y,x], then [seg id, zmin, zmax, etc]
    bboxes = simple_chunk(None, [vol], chunk_size, False, True, _chunk_bbox)
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
    return np.array(result)


def _chunk_cc3d_neighbors(
    z,
    y,
    x,
    chunk_size,
    partial_cc3d,
    partial_statistics,
    uf,
    connectivity,
    remapping,
    group_cache,
    mask,
):
    # performs unions on connected components based on neighbors
    mask = mask[
        : partial_cc3d.shape[0], : partial_cc3d.shape[1], : partial_cc3d.shape[2]
    ]
    # arange instead of zeros, to accomodate non-boundary components
    remapping[(z, y, x)] = np.arange(
        partial_statistics[z, y, x]["voxel_counts"].shape[0], dtype=partial_cc3d.dtype
    )

    # required for non-boundary components
    for i in range(1, partial_statistics[z, y, x]["voxel_counts"].shape[0]):
        uf.add((z, y, x, i))

    neighbors = group_cache.get(f"{z},{y},{x}")[:]
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

    unique_idx = get_is_first_unique(old_cc_idx)
    for i, is_first_unique in enumerate(unique_idx):
        if is_first_unique:
            continue
        uf.union(tuple(result[i]), tuple(result[i - 1]))


def _chunk_remap_cc3d(z, y, x, chunk_size, partial_cc3d, remapping):
    return remapping[(z, y, x)][partial_cc3d]


def _chunk_half_extend_cc3d(
    z, y, x, chunk_size, vol, zyx_idx, mask, group_cache, connectivity
):
    connected_components = cc3d.connected_components(
        vol != 0, connectivity=connectivity
    )
    # trim mask to fit chunk
    mask = mask[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    neighbors = np.concatenate(
        (
            zyx_idx[mask],
            connected_components[mask].reshape(-1, 1),
        ),
        axis=-1,
    ).reshape(-1, zyx_idx.shape[-1] + 1)
    dataset_neighbors = group_cache.create_dataset(
        f"{z},{y},{x}", neighbors.shape, dtype=neighbors.dtype
    )
    dataset_neighbors[:] = neighbors

    # will be trimmed automatically
    return connected_components


def chunk_cc3d(dataset_output, vol, group_cache, chunk_size, connectivity):
    # perform chunked cc3d calculations
    # dataset_output: output dataset to write to; if None, will write to nested array
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # dataset_cache: vol, 3
    # NOTE: if IndexError is emitted, result is invalid

    # cc3d only handles inputs > 1
    assert all([i > 1 for i in chunk_size])
    dataset_cache = group_cache.create_dataset(
        "cache", (*vol.shape, 3), dtype=vol.dtype
    )
    zyx_idx = simple_chunk(
        dataset_cache,
        vol.shape,
        chunk_size,
        False,
        True,
        lambda z, y, x, chunk_size: np.array([z, y, x]).reshape(1, 1, -1),
    )

    # mask for half extend
    mask = np.ones([i + 1 for i in chunk_size], dtype=bool)
    # 2 instead of 1 because of [[0,1], [1,0]] edge case
    mask[:-2, :-2, :-2] = False

    # perform cc3d on individual chunks
    partial_cc3d = simple_chunk(
        dataset_output,
        [vol, zyx_idx],
        chunk_size,
        "half_extend",
        True,
        _chunk_half_extend_cc3d,
        mask,
        group_cache,
        connectivity,
    )

    # TODO: don't hardcode dtype
    # get voxel_counts for each chunk
    partial_statistics = simple_chunk(
        None,
        [partial_cc3d],
        chunk_size,
        False,
        False,
        lambda vol: cc3d.statistics(vol.astype(np.uint64)),
    )

    # get adjacencies from neighboring chunks also initialize remapping
    uf = UnionFind()
    remapping = {}
    simple_chunk(
        None,
        [partial_cc3d],
        chunk_size,
        "half_extend",
        True,
        _chunk_cc3d_neighbors,
        partial_statistics,
        uf,
        connectivity,
        remapping,
        group_cache,
        mask,
    )

    # list of sets containing components
    mapping = np.array(uf.components(), dtype=object)
    # include 0 in voxel_counts
    voxel_counts = np.zeros(mapping.shape[0], dtype=int)
    for i in range(mapping.shape[0]):
        for val in mapping[i]:
            voxel_statistics = partial_statistics[val[0], val[1], val[2]][
                "voxel_counts"
            ]
            # due to half_extend_cc3d connections which are not included in statistics
            if val[3] < voxel_statistics.shape[0]:
                voxel_counts[i] += voxel_statistics[val[3]]
    order = np.argsort(voxel_counts)[::-1]
    voxel_counts = voxel_counts[order]
    mapping = mapping[order]

    voxel_counts = np.concatenate(
        (
            [vol.shape[0] * vol.shape[1] * vol.shape[2] - np.sum(voxel_counts)],
            voxel_counts,
        )
    )

    voxel_counts = voxel_counts[: np.max(np.nonzero(voxel_counts)[0]) + 1]
    assert np.all(voxel_counts[1:] > 0)

    # NOTE: can vectorize remapping assignment
    for i, key in enumerate(range(mapping.shape[0]), 1):
        for val in mapping[key]:
            remapping[(val[0], val[1], val[2])][val[3]] = i

    # remap dataset_output
    simple_chunk(
        partial_cc3d,
        [partial_cc3d],
        chunk_size,
        False,
        True,
        _chunk_remap_cc3d,
        remapping,
    )
    return partial_cc3d, voxel_counts


if __name__ == "__main__":
    # file = h5py.File("./den_ruilin_v2_16nm.h5").get("main")
    chunk_size = (100, 100, 100)
    idx = 1
    file = h5py.File("./den_s24_16nm.h5").get("main")

    output = h5py.File("test_single_idx.hdf5", "w")
    output.create_dataset("single_idx", file.shape, dtype="b")
    output.create_dataset("cc3d_output", file.shape, dtype="uint16")
    single_idx = output.get("single_idx")
    cc3d_output = output.get("cc3d_output")

    simple_chunk(single_idx, [file], chunk_size, False, False, lambda vol: vol == idx)
    chunk_cc3d(cc3d_output, single_idx, chunk_size, connectivity=26)
