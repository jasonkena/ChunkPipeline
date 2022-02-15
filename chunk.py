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
    # half: whether only to extend right side
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
    dataset_output, dataset_inputs, chunk_size, pad, func, *args, **kwargs
):
    # chunks operations (func) on input
    # dataset_output: output dataset to write to; if None, will write to nested array
    # dataset_inputs: list of h5 datasets
    # chunk_size: [size_z, size_y, size_x]
    # pad: "zero", "extend", or False value, output will be trimmed
    # zero will always pad each dimension by 2, extend will do the same if not on boundary
    # func, *args, **kwargs: self-explanatory

    # NOTE: assumes input sizes are all equal and that output size = input size

    if dataset_output == None:
        assert pad == False
        dataset_output = []
    shape = dataset_inputs[0].shape
    n_iter = [math.ceil(shape[i] / chunk_size[i]) for i in range(3)]

    for z in range(n_iter[0]):
        for y in range(n_iter[1]):
            for x in range(n_iter[2]):
                slices = np.s_[
                    z * chunk_size[0] : (z + 1) * chunk_size[0],
                    y * chunk_size[1] : (y + 1) * chunk_size[1],
                    x * chunk_size[2] : (x + 1) * chunk_size[2],
                ]
                if pad == "extend":
                    slices, shrink_slices = extend_slices(slices, False)

                inputs = [i[slices] for i in dataset_inputs]
                if pad == "zero":
                    inputs = [pad_vol(i, [3, 3, 3]) for i in inputs]

                output = func(*inputs, *args, *kwargs)

                if pad == "zero":
                    output = output[1:-1, 1:-1, 1:-1]
                elif pad == "extend":
                    output = output[shrink_slices]

                if isinstance(dataset_output, list):
                    dataset_output.append(output)
                else:
                    dataset_output[slices] = output

    if isinstance(dataset_output, list):
        dataset_output = np.array(dataset_output, dtype=object)
        return dataset_output.reshape(*n_iter)
    return dataset_output

def get_is_first_unique(array):
    # assuming sorted, return whether an element is the first unique element
    # assuming no inputs are equal to 0
    return np.pad(array[:-1], (1, 0)) != array

def chunk_bbox(vol, chunk_size):
    # calculate bbox in chunks
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # returns bboxes in form [seg id, zmin, zmax, etc]

    # restore original coordinates
    # [z,y,x], then [seg id, zmin, zmax, etc]
    bboxes = simple_chunk(None, [vol], chunk_size, False, get_bb_all3d)
    for z in range(bboxes.shape[0]):
        for y in range(bboxes.shape[1]):
            for x in range(bboxes.shape[2]):
                bboxes[z, y, x][:, 1:3] += chunk_size[0] * z
                bboxes[z, y, x][:, 3:5] += chunk_size[1] * y
                bboxes[z, y, x][:, 5:7] += chunk_size[2] * x

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


def shrink_slices(slices, shrink, idx):
    # shrink: [shrink_z, shrink_y, shrink_x] whether to compress dimension
    # idx: [z, y, x] cluster coords
    # returns overall slice, original part of slice, other part of slice, along with idx of other part of slice
    new_slices = [
        [slice(slices[i].stop - 1, slices[i].stop + 1), slice(0, 1), slice(1, 2)]
        if shrink[i]
        else [slices[i], slice(None, None, None), slice(None, None, None)]
        for i in range(len(slices))
    ]
    new_slices = np.array(new_slices, dtype=object).T.tolist()
    new_slices = [tuple(i) for i in new_slices]

    new_idx = [idx[i] + shrink[i] for i in range(len(slices))]
    return [*new_slices, new_idx]


def neighbor_slices(slices, idx):
    # generates 7 slices (each with 3 slices and idx)
    # 3 for faces, 3 for edges, 1 for corner

    # skipping idx 0 which is doesn't shrink anything
    shrinks = list(itertools.product([0, 1], repeat=3))[1:]
    all_slices = [shrink_slices(slices, shrink, idx) for shrink in shrinks]
    return all_slices


def chunk_cc3d(dataset_output, vol, chunk_size, connectivity):
    # perform chunked cc3d calculations
    # dataset_output: output dataset to write to; if None, will write to nested array
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]

    partial_cc3d = simple_chunk(
        dataset_output,
        [vol],
        chunk_size,
        False,
        cc3d.connected_components,
        connectivity=connectivity,
    )
    partial_statistics = simple_chunk(
        None, [partial_cc3d], chunk_size, False, cc3d.statistics
    )

    # merge components from adjacent chunks
    uf = UnionFind()
    shape = partial_cc3d.shape
    n_iter = [math.ceil(shape[i] / chunk_size[i]) for i in range(3)]

    for z in range(n_iter[0]):
        for y in range(n_iter[1]):
            for x in range(n_iter[2]):
                # required for non-boundary components
                for i in range(1, partial_statistics[z][y][x].voxel_counts.shape[0]+1):
                    uf.add((z, y, x, i))

                slices = np.s_[
                    z * chunk_size[0] : (z + 1) * chunk_size[0],
                    y * chunk_size[1] : (y + 1) * chunk_size[1],
                    x * chunk_size[2] : (x + 1) * chunk_size[2],
                ]
                neighboring_slices = neighbor_slices(slices, [z, y, x])

                # [# neighbors, shape]
                full_slices = [partial_cc3d[i[0]] for i in neighboring_slices]

                # [# neighbors, shape]
                cc_slices = [
                    cc3d.connected_components(
                        i != 0,
                        connectivity=connectivity,
                    )
                    for i in full_slices
                ]

                # iterate over neighbors
                for i in range(len(cc_slices)):
                    cc_idx = np.concatenate(
                        (
                            cc_slices[i][neighboring_slices[i][1]].reshape(-1),
                            cc_slices[i][neighboring_slices[i][2]].reshape(-1),
                        )
                    )
                    idx_order = np.argsort(cc_idx)
                    cc_idx = cc_idx[idx_order]


                    base_idx = np.concatenate(
                        (
                            np.stack(
                                np.broadcast_arrays(
                                    z,
                                    y,
                                    x,
                                    full_slices[i][neighboring_slices[i][1]].reshape(
                                        -1
                                    ),
                                ),
                                axis=1,
                            ),
                            np.stack(
                                np.broadcast_arrays(
                                    *neighboring_slices[i][3], # z,y,x of neighbor
                                    full_slices[i][neighboring_slices[i][2]].reshape(-1)
                                ),
                                axis=1,
                            ),
                        )
                    )
                    base_idx = base_idx[cc_idx]

                    is_valid = cc_idx != 0
                    cc_idx = cc_idx[is_valid]
                    base_idx = base_idx[is_valid]

                    unique_idx = get_is_first_unique(cc_idx)

                    for j, is_first_unique in enumerate(unique_idx):
                        if is_first_unique:
                            continue
                        uf.union(tuple(base_idx[j]), tuple(base_idx[j-1]))

    mapping = uf.component_mapping()
    voxel_counts = {key: 0 for key in mapping.keys()}
    for key in mapping.keys():
        for val in mapping[key]:
            voxel_counts[key] += partial_statistics[val[0]][val[1]][val[2]].voxel_counts[val[3]]
    sorted_keys = sorted(mapping, key=lambda key: voxel_counts[key], reverse=True)
    voxel_counts = np.pad(np.array(sorted(voxel_counts.values(), reverse=True)), (1,0))

    remapping = {}
    for z in range(n_iter[0]):
        for y in range(n_iter[1]):
            for x in range(n_iter[2]):
                # arange instead of zeros, to accomodate non-boundary components
                remapping[(z,y,x)] = np.arange(partial_statistics[z][y][x].voxel_counts.shape[0])
    for i, key in enumerate(sorted_keys, 1):
        for val in mapping[key]:
            remapping[(val[0],val[1], val[2])][val[3]] = i

    def remap_chunk(chunk):
        chunk[:]

    # remap dataset_output
    simple_chunk(
        None,
        [dataset_output],
        chunk_size,
        False,
        lambda chunk: chunk # remap
    )

if __name__ == "__main__":
    a = np.random.rand(10, 10, 10)

    slices = np.s_[0:5, 2:3, 5:15]
    result = shrink_slices(slices, [True, False, False], [0, 0, 0])
