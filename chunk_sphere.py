import edt
import h5py
import torch
import torch.nn.functional as F
import math
import expand_parabola
import os
import sys
from settings import *


import numpy as np
import chunk
from utils import pad_vol
from settings import *

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


def _get_expand_edt(vol, anisotropy):
    if not vol.flags["C_CONTIGUOUS"]:
        vol = np.ascontiguousarray(vol)

    result = (
        expand_parabola.expand_edt(
            vol,
            anisotropy=anisotropy,
            order="C",  # was C
            parallel=0,  # max CPU
        )
        > 0
    )

    return [result]


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


def sphere_iteration(
    group_cache,
    expanded,
    dt,
    vol,
    threshold,
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

    pad_width = [math.ceil(threshold / i) for i in anisotropy]
    chunk.simple_chunk(
        [group_cache.create_dataset("new_expanded", expanded.shape, dtype=bool)],
        [expanded, dt, vol],
        chunk_size,
        lambda expanded, dt, vol: [
            np.logical_and(
                _get_expand_edt(expanded * (dt + threshold), anisotropy=anisotropy)[0],
                vol,
            )
        ],
        num_workers,
        pad="extend",
        pad_width=pad_width,
    )

    del group_cache["expanded"]
    group_cache.move("new_expanded", "expanded")

    return group_cache.get("expanded")


def extract(
    group_cache,
    vol,
    chunk_size,
    anisotropy,
    connectivity,
    max_erode,
    erode_delta,
    num_iter,
    num_workers,
):
    # gets volume segmentation
    # vol: binary 3d volume
    # max_erode: int/float, thresholding distance to cut off spines
    # erode_delta: extra dilation over max_erode
    # anisotropy: [z-size, y-size, x-size]
    # connectivity: read cc3d docs

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
    expanded, largest_voxel_counts = chunk.chunk_cc3d(
        group_cache.create_dataset("expanded", remaining.shape, dtype="uint16"),
        remaining,
        group_cache,
        chunk_size,
        connectivity,
        num_workers,
        k=1,
    )

    # TODO: assert that final segmentation is only composed of single CC
    for _ in range(num_iter):
        expanded = sphere_iteration(
            group_cache,
            expanded,
            dt,
            vol,
            max_erode + erode_delta,
            anisotropy,
            chunk_size,
            num_workers,
        )

    others = chunk.simple_chunk(
        [group_cache.create_dataset("others", shape=expanded.shape, dtype=bool)],
        [vol, expanded],
        chunk_size,
        lambda vol, expanded: [np.logical_xor(vol, expanded)],
        num_workers,
    )
    # segment the non trunks
    cc3d_others, voxel_counts = chunk.chunk_cc3d(
        group_cache.create_dataset("cc3d_others", others.shape, dtype="uint16"),
        others,
        group_cache,
        chunk_size,
        connectivity,
        num_workers,
        k=False,
    )
    voxel_counts = np.concatenate([largest_voxel_counts[1:], voxel_counts[1:]])
    voxel_counts = np.concatenate([[vol.shape[0]*vol.shape[1]*vol.shape[2]-np.sum(voxel_counts)], voxel_counts])

    print(f"voxel_counts: {voxel_counts}")

    seg = chunk.simple_chunk(
        [group_cache.create_dataset("seg", others.shape, dtype="uint16")],
        [expanded, cc3d_others],
        chunk_size,
        # relabel so that trunk is idx 1
        lambda expanded, cc3d_others: [cc3d_others + (cc3d_others > 0) + expanded],
        num_workers,
    )

    return seg


def main(input_path, id):
    if os.path.exists(os.path.join("baseline", f"{str(id)}.h5")):
        return

    input = h5py.File(os.path.join(input_path, f"{str(id)}.h5"))
    output = h5py.File(os.path.join(input_path, f"seg_{str(id)}.h5"), "w")
    # no need to create actual group
    group_cache = output

    extract(
        group_cache,
        input.get("main"),
        CHUNK_SIZE,
        ANISOTROPY,
        CONNECTIVITY,
        MAX_ERODE,
        ERODE_DELTA,
        NUM_ITER,
        NUM_WORKERS,
    )
    for key in group_cache.keys():
        if key != "seg":
            del group_cache[key]
    output.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
