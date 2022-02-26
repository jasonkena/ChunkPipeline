import numpy as np
import h5py
from utils import pad_vol
from imu.io import get_bb_all3d
from unionfind import UnionFind
import itertools
import cc3d
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

import math


def extend_slices(slices, half, pad_width):
    # computes slices to extend slice to include 1 neighboring voxel, and corresponding slice to undo it, restoring original size
    # slices: input slices from np.s_
    # half: whether to only extend right side
    # returns new_slices and shrink_slices
    new_slices = tuple(
        slice(
            max(0, slices[i].start - (not half) * pad_width[i]),
            slices[i].stop + pad_width[i],
        )
        for i in range(len(slices))
    )
    left_start = [
        (pad_width[i] if slices[i].start >= pad_width[i] else slices[i].start)
        * (not half)
        for i in range(len(slices))
    ]
    distances = [i.stop - i.start for i in slices]

    shrink_slices = tuple(
        slice(
            left_start[i],
            left_start[i] + distances[i],
        )
        for i in range(len(slices))
    )

    return new_slices, shrink_slices


class ChunkDataset(Dataset):
    def __init__(self, dataset_inputs, zyx_idx, chunk_size, pad, pad_width):
        self.dataset_inputs = dataset_inputs
        self.zyx_idx = zyx_idx
        self.chunk_size = chunk_size
        self.pad = pad
        self.pad_width = pad_width

    def __len__(self):
        return len(self.zyx_idx)

    def __getitem__(self, idx):
        # TODO: rewrite func to enable asynchronous processing (e.g., UnionFind merging), now limited to doing multiprocessing on input
        z, y, x = self.zyx_idx[idx]
        original_slices = np.s_[
            z * self.chunk_size[0] : (z + 1) * self.chunk_size[0],
            y * self.chunk_size[1] : (y + 1) * self.chunk_size[1],
            x * self.chunk_size[2] : (x + 1) * self.chunk_size[2],
        ]
        if self.pad == "extend" or self.pad == "half_extend":
            slices, shrink_slices = extend_slices(
                original_slices, self.pad == "half_extend", self.pad_width
            )
        else:
            slices = original_slices
            shrink_slices = None

        if isinstance(self.dataset_inputs, list):
            inputs = [i[slices] for i in self.dataset_inputs]
        else:
            inputs = []
        if self.pad == "zero":
            inputs = [pad_vol(i, [j * 2 + 1 for j in self.pad_width]) for i in inputs]
        return {
            "z": z,
            "y": y,
            "x": x,
            "inputs": inputs,
            "shrink_slices": shrink_slices,
            "original_slices": original_slices,
        }


def simple_chunk(
    dataset_outputs,
    dataset_inputs,
    chunk_size,
    func,
    num_workers,
    pad=False,
    pass_params=False,
    bbox=False,
    pad_width=(1, 1, 1),
    *args,
    **kwargs,
):

    # chunks operations (func) on input
    # dataset_outputs: output datasets to write to; if None, will write to nested array
    # dataset_inputs: list of h5 datasets; if non-list, will be treated as size of input for zyx
    # chunk_size: [size_z, size_y, size_x]
    # pad: "zero", "extend", "half_extend" or False value, output will be trimmed
    # zero will always pad each dimension by 2, extend will do the same if not on boundary, half_extend right_extends
    # func, *args, **kwargs: self-explanatory
    # bbox: inclusive ranges to perform the computation on, of the form [id, z1, z2, y1, y2, x1, x2]

    # NOTE: assumes input sizes are all equal and that output size = input size
    # NOTE: func should not overwrite input; just pass same dataset as output

    assert pad in ["zero", "extend", "half_extend", False]
    if isinstance(dataset_inputs, list):
        # TODO: verify all inputs
        shape = dataset_inputs[0].shape
    else:
        shape = dataset_inputs
    dataset_outputs = [i if i is not None else [] for i in dataset_outputs]
    for dataset_output in dataset_outputs:
        if not isinstance(dataset_output, list):
            assert dataset_output.shape[: len(shape)] == shape

    if bbox is not False:
        bbox = bbox[1:]
        ranges = [
            # takes into account bbox's inclusivity
            range(
                math.floor(bbox[2 * i] / chunk_size[i]),
                math.floor(bbox[2 * i + 1] / chunk_size[i]) + 1,
            )
            for i in range(3)
        ]
    else:
        ranges = [range(math.ceil(shape[i] / chunk_size[i])) for i in range(3)]

    zyx_idx = []
    for z in ranges[0]:
        for y in ranges[1]:
            for x in ranges[2]:
                zyx_idx.append([z, y, x])

    chunk_dataset = ChunkDataset(dataset_inputs, zyx_idx, chunk_size, pad, pad_width)
    # defaults to prefetching 2*num_workers; do not collate
    chunk_dataloader = DataLoader(
        chunk_dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x[0]
    )
    for inputs in tqdm(chunk_dataloader):
        if pass_params:
            params = {
                "z": inputs["z"],
                "y": inputs["y"],
                "x": inputs["x"],
                "chunk_size": chunk_size,
                "shrink_slices": inputs["shrink_slices"],
            }
            output = func(
                params,
                *inputs["inputs"],
                *args,
                **kwargs,
            )
        else:
            output = func(*inputs["inputs"], *args, **kwargs)

        if len(dataset_outputs):
            assert len(output) == len(dataset_outputs)

        for i in range(len(dataset_outputs)):
            if not isinstance(dataset_outputs[i], list):
                if pad == "zero":
                    output[i] = output[i][
                        pad_width[0] : -pad_width[0],
                        pad_width[1] : -pad_width[1],
                        pad_width[2] : -pad_width[2],
                    ]
                elif pad == "extend" or pad == "half_extend":
                    output[i] = output[i][inputs["shrink_slices"]]
                dataset_outputs[i][inputs["original_slices"]] = output[i]
            else:
                dataset_outputs[i].append(output[i])

    for i in range(len(dataset_outputs)):
        if isinstance(dataset_outputs[i], list):
            dataset_output = np.array(dataset_outputs[i], dtype=object)
            shape = [len(i) for i in ranges]
            if dataset_output.ndim > 1:
                shape.append(-1)
            dataset_outputs[i] = dataset_output.reshape(shape)
    if len(dataset_outputs) == 1:
        return dataset_outputs[0]
    return dataset_outputs


def get_is_first_unique(array):
    # assuming sorted, return whether an element is the first unique element
    assert np.all(array != 0)
    return np.pad(array[:-1], (1, 0)) != array


def _chunk_bbox(params, vol):
    z, y, x, chunk_size = [params[i] for i in ["z", "y", "x", "chunk_size"]]
    # add offset to bounding boxes based on zyx
    bboxes = get_bb_all3d(vol)
    bboxes[:, 1:3] += chunk_size[0] * z
    bboxes[:, 3:5] += chunk_size[1] * y
    bboxes[:, 5:7] += chunk_size[2] * x

    return [bboxes]


def chunk_bbox(vol, chunk_size, num_workers):
    # calculate bbox in chunks
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # returns bboxes in form [seg id, zmin, zmax, etc]

    # restore original coordinates
    # [z,y,x], then [seg id, zmin, zmax, etc]
    bboxes = simple_chunk(
        [None], [vol], chunk_size, _chunk_bbox, num_workers, pass_params=True
    )
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
    params,
    partial_cc3d,
    partial_statistics,
    uf,
    remapping,
    group_cache,
    mask,
):
    # NOTE: will need to rewrite for parallelism, remove side effects
    z, y, x = [params[i] for i in ["z", "y", "x"]]
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


def _chunk_remap_cc3d(params, partial_cc3d, remapping):
    z, y, x = [params[i] for i in ["z", "y", "x"]]
    return [remapping[(z, y, x)][partial_cc3d]]


def _chunk_half_extend_cc3d(params, vol, zyx_idx, mask, group_cache, connectivity):
    z, y, x = [params[i] for i in ["z", "y", "x"]]
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
    return [connected_components]


def chunk_cc3d(
    dataset_output, vol, group_cache, chunk_size, connectivity, num_workers, k
):
    # perform chunked cc3d calculations
    # dataset_output: output dataset to write to; if None, will write to nested array
    # vol: h5 dataset used as input
    # chunk_size: [size_z, size_y, size_x]
    # dataset_cache: vol, 3
    # k: either int or False
    # NOTE: if IndexError is emitted, result is invalid

    # cc3d only handles inputs > 1
    assert all([i > 1 for i in chunk_size])
    dataset_cache = group_cache.create_dataset(
        "cache", (*vol.shape, 3), dtype=vol.dtype
    )
    zyx_idx = simple_chunk(
        [dataset_cache],
        vol.shape,
        chunk_size,
        lambda params: [
            np.array([params["z"], params["y"], params["x"]]).reshape(1, 1, -1)
        ],
        num_workers,
        pass_params=True,
    )

    # mask for half extend
    mask = np.ones([i + 1 for i in chunk_size], dtype=bool)
    # 2 instead of 1 because of [[0,1], [1,0]] edge case
    mask[:-2, :-2, :-2] = False

    # perform cc3d on individual chunks
    partial_cc3d = simple_chunk(
        [dataset_output],
        [vol, zyx_idx],
        chunk_size,
        _chunk_half_extend_cc3d,
        num_workers,
        pad="half_extend",
        pass_params=True,
        mask=mask,
        group_cache=group_cache,
        connectivity=connectivity,
    )

    # TODO: don't hardcode dtype
    # get voxel_counts for each chunk
    partial_statistics = simple_chunk(
        [None],
        [partial_cc3d],
        chunk_size,
        lambda vol: [cc3d.statistics(vol.astype(np.uint64))],
        num_workers,
    )

    # get adjacencies from neighboring chunks also initialize remapping
    uf = UnionFind()
    remapping = {}
    simple_chunk(
        [],
        [partial_cc3d],
        chunk_size,
        _chunk_cc3d_neighbors,
        num_workers,
        pad="half_extend",
        pass_params=True,
        partial_statistics=partial_statistics,
        uf=uf,
        remapping=remapping,
        group_cache=group_cache,
        mask=mask,
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
            if k:
                remapping[(val[0], val[1], val[2])][val[3]] = i * (i <= k)
            else:
                remapping[(val[0], val[1], val[2])][val[3]] = i

    # remap dataset_output
    simple_chunk(
        [partial_cc3d],
        [partial_cc3d],
        chunk_size,
        _chunk_remap_cc3d,
        num_workers,
        pass_params=True,
        remapping=remapping,
    )

    if k:
        voxel_counts[0] += np.sum(voxel_counts[k + 1 :])
        voxel_counts = voxel_counts[: k + 1]
    return partial_cc3d, voxel_counts


def _chunk_argwhere(params, *args, **kwargs):
    z, y, x, chunk_size = [params[i] for i in ["z", "y", "x", "chunk_size"]]
    # kwargs must contain func, and it must return 2 things: binary mask and another (array or None)
    # returns indices where vol is true
    new_kwargs = kwargs.copy()
    func = new_kwargs.pop("chunk_func")
    mask, extra = func(params, *args, **new_kwargs)

    idx = np.argwhere(mask)
    if extra is not None:
        extra_dim = 1 if mask.ndim < 4 else mask.shape[3]
        idx = np.concatenate((idx, extra[mask].reshape(-1, extra_dim)), axis=1)
    idx[:, 0] += chunk_size[0] * z
    idx[:, 1] += chunk_size[1] * y
    idx[:, 2] += chunk_size[2] * x

    return [idx]


def chunk_argwhere(dataset_inputs, chunk_size, chunk_func, pad, num_workers):
    # TODO: implement chunked saving instead of aggregating all indices
    return np.concatenate(
        simple_chunk(
            [None],
            dataset_inputs,
            chunk_size,
            _chunk_argwhere,
            num_workers,
            pad=pad,
            pass_params=True,
            chunk_func=chunk_func,
        ).reshape(-1)
    )


def _chunk_unique(params, vol, return_inverse):
    z, y, x = [params[i] for i in ["z", "y", "x"]]
    shape = vol.shape

    output = np.unique(vol, return_inverse=return_inverse)

    if not return_inverse:
        return [output]

    return [
        [z, y, x, output[0]],
        output[1].reshape(shape),
    ]


def chunk_unique(dataset_input, chunk_size, return_inverse, num_workers):
    # return_inverse: either None or equal
    # NOTE: will need to rewrite if number of unique values exceed memory
    # simple_chunk(dataset_input, )
    output = simple_chunk(
        [None] if return_inverse is None else [None, return_inverse],
        [dataset_input],
        chunk_size,
        _chunk_unique,
        num_workers,
        pass_params=True,
        return_inverse=(return_inverse is not None),
    )
    if return_inverse is None:
        unique = output
    else:
        unique, idx = output

    # TODO: remove explicit dimension usage
    if return_inverse is None:
        # output unique might be (z,y,x, -1) or just (z,y,x, dtype=object)
        if unique.ndim > 3:
            final_unique = np.unique(unique.reshape(-1))
        else:
            final_unique = np.unique(np.concatenate(unique.reshape(-1)))
        return final_unique

    flattened = unique.reshape(-1, 4)
    final_unique = np.unique(np.concatenate(flattened[:, 3]))
    remapping = {}
    for row in flattened:
        # NOTE: might be more efficient to implement own searchsorted using np.unique, since all inputs are sorted
        remapping[tuple(row[:3].tolist())] = np.searchsorted(final_unique, row[3])
    __import__("pdb").set_trace()

    # for i in range()

    # NOTE: need to remap input idx, cann't remap remappiung

    # TODO: don't hardcode dtype
    max_unique = final_unique.max()
    remapping = np.zeros(max_unique + 1, dtype=np.uint64)
    remapping[final_unique] = np.arange(max_unique)


def _chunk_write_seg(params, vol, output, bbox_in, filter_id):
    z, y, x, chunk_size = [params[i] for i in ["z", "y", "x", "chunk_size"]]
    # NOTE: will need to rewrite this to implement parallelism
    idx = [z, y, x]
    # extra +1 due to bbox format
    read_slices = [
        slice(
            max(0, bbox_in[2 * i + 1] - idx[i] * chunk_size[i]),
            min(chunk_size[i] - 1, bbox_in[2 * i + 1 + 1] - idx[i] * chunk_size[i]),
        )
        for i in range(3)
    ]
    distances = [i.stop - i.start for i in read_slices]
    write_slices = [
        slice(
            max(idx[i] * chunk_size[i] - bbox_in[2 * i + 1], 0),
            max(idx[i] * chunk_size[i] - bbox_in[2 * i + 1], 0) + distances[i],
        )
        for i in range(3)
    ]
    if filter_id:
        vol = vol == bbox_in[0]
    output[
        write_slices[0].start : write_slices[0].stop + 1,
        write_slices[1].start : write_slices[1].stop + 1,
        write_slices[2].start : write_slices[2].stop + 1,
    ] = vol[
        read_slices[0].start : read_slices[0].stop + 1,
        read_slices[1].start : read_slices[1].stop + 1,
        read_slices[2].start : read_slices[2].stop + 1,
    ]


def get_seg(output, vol, bbox, chunk_size, filter_id, num_workers):
    # NOTE: need to reimplement this for func parallelism
    # extra +1 due to bbox format
    # shape = (1 + bbox[2 * i + 1 + 1] - bbox[2 * i + 1] for i in range(3))

    simple_chunk(
        [],
        [vol],
        chunk_size,
        _chunk_write_seg,
        num_workers,
        pass_params=True,
        output=output,
        bbox=bbox,
        bbox_in=bbox,
        filter_id=filter_id,
    )
    return output


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

    simple_chunk(single_idx, [file], chunk_size, lambda vol: vol == idx)
    chunk_cc3d(cc3d_output, single_idx, chunk_size, connectivity=26)
