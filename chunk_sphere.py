import cc3d
import edt
import h5py
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from imu.io import get_bb_all3d
import os


import numpy as np
import chunk
from utils import pad_vol
from settings import *
import point

# NOTE: here naive separation between segments is used: each segment id is processed separately
# can potentially come up with a way to do it in a chunk-based manner instead of by segments
# will need to deal with chunk boundaries then


def _chunk_get_boundary(vol):
    vol = vol > 0
    padded_vol = torch.from_numpy(pad_vol(vol, [3, 3, 3]))
    vol = torch.from_numpy(vol)
    boundary = torch.logical_and(
        F.max_pool3d((~padded_vol).float().unsqueeze(0), kernel_size=3, stride=1), vol
    ).squeeze(0)
    return [boundary.numpy()]


def get_boundary(boundary_dataset, vol, chunk_size, num_workers):
    # gets foreground voxels which "touch" background pixels, as defined by a 3x3x3 kernel
    # vol: 3d volume (with 0 indicating background)
    # NOTE: this function intentionally ignores anisotropy
    # TODO: can prevent double input chunks to full chunk_size; no way to elegantly implement it

    return chunk.simple_chunk(
        [boundary_dataset],
        [vol],
        chunk_size,
        _chunk_get_boundary,
        num_workers,
        pad="extend",
    )


def _get_dt(vol, anisotropy, black_border):
    # NOTE: might need to pad by one for chunks on borders of input volume

    if not vol.flags["C_CONTIGUOUS"]:
        vol = np.ascontiguousarray(vol)

    dt = edt.edt(
        vol,
        anisotropy=anisotropy,
        black_border=black_border,
        order="C",  # was C
        parallel=0,  # max CPU
    )
    assert not np.isnan(dt).any()

    return [dt]


def get_dt(
    dataset_output,
    vol,
    chunk_size,
    anisotropy,
    black_border,
    threshold,
    num_workers,
):
    # computes euclidean distance transform (voxel-wise distance to nearest background)
    # vol: 3d volume (with 0 indicating back)
    # anisotropy: [z-size, y-size, x-size]
    # black_border: whether volume boundaries should be treated as foreground or background
    # threshold: edt distance threshold
    pad_width = [math.ceil(threshold / i) for i in anisotropy]

    return chunk.simple_chunk(
        [dataset_output],
        [vol],
        chunk_size,
        _get_dt,
        num_workers,
        pad="extend",
        pass_params=False,
        pad_width=pad_width,
        anisotropy=anisotropy,
        black_border=black_border,
    )


def get_sphere_bounds(boundary_idx, vals, boundary_inverse, erode_delta, anisotropy):
    # computes bounding boxes for every boundary edt value
    # returns [[zmin, zmax], [ymin, ymax], [xmin, xmax]] for every val
    # boundary_idx: [z_i, y_i, x_i], result from np.nonzero(boundary)
    # vals: edt values from boundary
    # boundary_inverse: 1d array, mapping from each nonzero to vals
    # erode_delta: nm, how much to dilate beyond edt
    # anisotropy: [z-size, y-size, x-size]
    result = []

    # iterate over unique edt values
    for i in range(len(vals)):
        # z,y,x nonzero indices with specific edt value
        coords = [boundary_idx[j][boundary_inverse == i] for j in range(3)]
        # NOTE: may need to add +1 for good measure
        # sphere radius in terms of voxels
        offsets = [math.ceil((vals[i] + erode_delta) / anisotropy[j]) for j in range(3)]

        # bounding boxes for each sphere radius
        result.append(
            [
                [max(0, np.min(coords[j]) - offsets[j]), np.max(coords[j]) + offsets[j]]
                for j in range(3)
            ]
        )
    return result


def sphere_expansion(vals, dt_boundary, sphere_bounds, erode_delta, anisotropy):
    # expand spheres given inputs
    # vals: edt values from boundary
    # dt_boundary: dt from bg * boundary
    # sphere_bounds: bounding box for each sphere radius
    # erode_delta: nm, how much to dilate beyond edt
    # anisotropy: [z-size, y-size, x-size]
    # returns expanded spheres

    result = np.zeros_like(dt_boundary, dtype=bool)
    print("Expanding spheres")
    for i in tqdm(range(len(vals))):
        zs, ys, xs = sphere_bounds[i]
        subvol = (
            dt_boundary[zs[0] : zs[1] + 1, ys[0] : ys[1] + 1, xs[0] : xs[1] + 1]
        ) != vals[i]
        # distance from center of spheres
        dt = get_dt(subvol, anisotropy, black_border=False)
        result[zs[0] : zs[1] + 1, ys[0] : ys[1] + 1, xs[0] : xs[1] + 1] = np.logical_or(
            result[zs[0] : zs[1] + 1, ys[0] : ys[1] + 1, xs[0] : xs[1] + 1],
            dt <= (vals[i] + erode_delta),
        )
    return result


def sphere_iteration(
    group_cache,
    expanded,
    dt,
    vol,
    erode_delta,
    anisotropy,
    chunk_size,
    num_workers,
):
    # performs sphere_iteration on volumes that may already be dilated
    # dt: edt from background in original volume
    # vol: original volume
    # erode_delta: nm, how much to dilate beyond edt
    # anisotropy: [z-size, y-size, x-size]
    # returns newly dilated volume

    boundary = get_boundary(
        group_cache.create_dataset("boundary", expanded.shape, dtype=bool),
        expanded,
        chunk_size,
        num_workers,
    )
    boundary_idx = chunk.chunk_argwhere(
        [boundary],
        chunk_size,
        lambda params, vol: [vol, None],
        False,
        num_workers,
    )

    vals, boundary_inverse = np.unique(dt[boundary], return_inverse=True)

    sphere_bounds = get_sphere_bounds(
        boundary_idx, vals, boundary_inverse, erode_delta, anisotropy
    )
    dt_boundary = dt * boundary
    expanded_sphere = sphere_expansion(
        vals, dt_boundary, sphere_bounds, erode_delta, anisotropy
    )
    expanded_sphere = np.logical_and(np.logical_or(expanded, expanded_sphere), vol)

    return expanded_sphere


def extract(
    group_cache,
    vol,
    chunk_size,
    anisotropy,
    connectivity,
    max_erode,
    erode_delta,
    num_iter,
    bbox,
    num_workers,
):
    # gets volume segmentation
    # vol: binary 3d volume
    # max_erode: int/float, thresholding distance to cut off spines
    # erode_delta: extra dilation over max_erode
    # anisotropy: [z-size, y-size, x-size]
    # connectivity: read cc3d docs

    print("Do get_seg")
    # raise ValueError
    # NOTE: assumes volume is connected

    dt = get_dt(
        group_cache.create_dataset("dt", vol.shape, dtype="f"),
        vol,
        chunk_size,
        anisotropy,
        black_border=False,
        threshold=max_erode + erode_delta,
        num_workers=num_workers,
    )
    remaining = chunk.simple_chunk(
        [group_cache.create_dataset("remaining", dt.shape, dtype=bool)],
        [dt],
        chunk_size,
        lambda dt: [dt >= max_erode],
        num_workers,
    )
    # TODO: do not hardcode dtype
    largest, _ = chunk.chunk_cc3d(
        group_cache.create_dataset("largest", remaining.shape, dtype="uint16"),
        remaining,
        group_cache,
        chunk_size,
        connectivity,
        num_workers,
        k=1,
    )
    # TODO: assert that final segmentation is only composed of single CC
    expanded = largest
    for _ in range(num_iter):
        expanded = sphere_iteration(None, expanded, dt, vol, erode_delta, anisotropy)

    others = np.logical_xor(vol, expanded)
    # segment the non trunks
    others, N_others = cc3d.connected_components(
        others, connectivity=connectivity, return_N=True
    )
    # relabel so that trunk is idx 1
    others[others > 0] += 1

    print(f"number of components in segmentation: {N_others+1}")
    seg = expanded + others

    return seg.astype(np.uint16)


def process_task(file, id, z1, z2, y1, y2, x1, x2):
    # given file, segmentation id, and seg bounding boxes, save segmentations
    # file: filename of original volume
    # id: segmentation id
    # ...: bbox

    save_file = os.path.join("results", f"{id}.npy")
    if os.path.isfile(save_file):
        return
    vol = h5py.File(file).get("main")
    assert vol.dtype == np.uint16

    # offset in order to fix dt on boundaries
    # new start
    nz, ny, nx = [max(0, i - 1) for i in [z1, y1, x1]]
    vol = vol[nz : z2 + 2, ny : y2 + 2, nx : x2 + 2]
    vol = np.ascontiguousarray(vol) == id

    seg = extract(vol, num_iter=1, max_erode=50, erode_delta=5, connectivity=26)
    # remove offset
    seg = seg[nz : nz + z2 - z1 + 1, ny : ny + y2 - y1 + 1, nx : nx + x2 - x1 + 1]
    np.save(save_file, seg)


if __name__ == "__main__":
    # file = h5py.File("./train-labels.h5").get("main")
    # file = h5py.File("./den_s24_16nm.h5").get("main")
    file = h5py.File("./den_ruilin_v2_16nm.h5").get("main")

    __import__("pdb").set_trace()
    vol = (np.array(file[:])).astype(np.uint16)
    alpha = np.unique(vol)
    # vol = (np.array(file[:]) == 1).astype(np.uint16)
    import time

    time0 = time.time()
    bbox = get_bbox(vol)
    time1 = time.time()
    print(time1 - time0)
    imu_bbox = get_bb_all3d(vol)
    time2 = time.time()
    print(time2 - time1)
    __import__("pdb").set_trace()
    small_vol = vol[
        bbox[0][0] : bbox[0][1] + 1,
        bbox[1][0] : bbox[1][1] + 1,
        bbox[2][0] : bbox[2][1] + 1,
    ]
    small_vol = np.ascontiguousarray(small_vol)

    seg = extract(small_vol, num_iter=2, max_erode=50, erode_delta=5, connectivity=26)
    np.save("seg.npy", seg)
