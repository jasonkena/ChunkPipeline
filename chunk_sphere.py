import edt
import h5py
import torch
import torch.nn.functional as F
import math
import os
import sys
from settings import *


import numpy as np
import chunk
from utils import pad_vol, create_compressed
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


def extract(
    group_cache,
    vol,
    chunk_size,
    anisotropy,
    connectivity,
    max_erode,
    erode_delta,
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
        create_compressed(group_cache, "dt", vol.shape, dtype="f"),
        vol,
        chunk_size,
        anisotropy,
        black_border=False,
        threshold=max_erode,
        num_workers=num_workers,
    )
    remaining = chunk.simple_chunk(
        [create_compressed(group_cache, "remaining", dt.shape, dtype=bool)],
        [dt],
        chunk_size,
        lambda dt: [dt >= max_erode],
        num_workers,
    )
    # TODO: do not hardcode dtype
    expanded, largest_voxel_counts = chunk.chunk_cc3d(
        create_compressed(group_cache, "expanded", remaining.shape, dtype="uint16"),
        remaining,
        group_cache,
        chunk_size,
        connectivity,
        num_workers,
        k=1,
    )
    inverted = chunk.simple_chunk(
        [create_compressed(group_cache, "inverted", expanded.shape, dtype=bool)],
        [expanded],
        chunk_size,
        lambda expanded: [expanded == 0],
        num_workers,
    )
    inverse_dt = get_dt(
        create_compressed(group_cache, "inverse_dt", inverted.shape, dtype="f"),
        inverted,
        chunk_size,
        anisotropy,
        black_border=False,
        threshold=max_erode + erode_delta,
        num_workers=num_workers,
    )
    trunk = chunk.simple_chunk(
        [create_compressed(group_cache, "trunk", inverse_dt.shape, dtype=bool)],
        [inverse_dt, vol],
        chunk_size,
        lambda inverse_dt, vol: [
            np.logical_and(inverse_dt <= max_erode + erode_delta, vol)
        ],
        num_workers,
    )

    others = chunk.simple_chunk(
        [create_compressed(group_cache, "others", shape=trunk.shape, dtype=bool)],
        [vol, trunk],
        chunk_size,
        lambda vol, trunk: [np.logical_xor(vol, trunk)],
        num_workers,
    )
    # segment the non trunks
    cc3d_others, voxel_counts = chunk.chunk_cc3d(
        create_compressed(group_cache, "cc3d_others", others.shape, dtype="uint16"),
        others,
        group_cache,
        chunk_size,
        connectivity,
        num_workers,
        k=False,
    )
    voxel_counts = np.concatenate([largest_voxel_counts[1:], voxel_counts[1:]])
    voxel_counts = np.concatenate(
        [
            [vol.shape[0] * vol.shape[1] * vol.shape[2] - np.sum(voxel_counts)],
            voxel_counts,
        ]
    )

    print(f"voxel_counts: {voxel_counts}")

    seg = chunk.simple_chunk(
        [create_compressed(group_cache, "seg", others.shape, dtype="uint16")],
        [trunk, cc3d_others],
        chunk_size,
        # relabel so that trunk is idx 1
        lambda trunk, cc3d_others: [cc3d_others + (cc3d_others > 0) + trunk],
        num_workers,
    )
    create_compressed(group_cache, "voxel_counts", data=voxel_counts)

    return seg


def main(base_path, id):
    if os.path.exists(os.path.join(base_path, "baseline", f"{str(id)}.h5")):
        return

    input = h5py.File(os.path.join(base_path, f"{str(id)}.h5"))
    output = h5py.File(os.path.join(base_path, f"seg_{str(id)}.h5"), "w")
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
        NUM_WORKERS,
    )
    for key in group_cache.keys():
        if key not in ["seg", "voxel_counts"]:
            del group_cache[key]

    create_compressed(
        group_cache,
        "seg_bbox",
        data=chunk.chunk_bbox(output.get("seg"), CHUNK_SIZE, NUM_WORKERS),
    )
    output.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
