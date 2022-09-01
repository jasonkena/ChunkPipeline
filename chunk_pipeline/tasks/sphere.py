import numpy as np
import edt
import h5py
from scipy import ndimage
import torch
import torch.nn.functional as F
import math

import chunk_pipeline.tasks.chunk as chunk
import expand_parabola

import dask.array as da
import dask

# NOTE: here naive separation between segments is used: each segment id is processed separately
# can potentially come up with a way to do it in a chunk-based manner instead of by segments
# will need to deal with chunk boundaries then


def _chunk_get_boundary(vol):
    return [
        ndimage.morphology.binary_dilation(
            ~np.pad(vol, ((1, 1), (1, 1), (1, 1))), structure=np.ones((3, 3, 3))
        )[1:-1, 1:-1, 1:-1]
    ]


# def _chunk_get_boundary(vol):
#     vol = torch.from_numpy(vol > 0)
#     # pad to guarantee that boundary inputs are also padded
#     padded_vol = F.pad(vol, (1, 1, 1, 1, 1, 1))
#     boundary = (
#         F.max_pool3d(
#             (~padded_vol.unsqueeze(0)).float(), kernel_size=3, stride=1
#         ).squeeze(0)
#         > 0
#     )
#     return [boundary.numpy()]
#


def get_boundary(vol):
    # gets foreground voxels which "touch" background pixels, as defined by a 3x3x3 kernel
    # vol: 3d volume (with 0 indicating background)
    # NOTE: this function intentionally ignores anisotropy
    # TODO: can prevent double input chunks to full chunk_size; no way to elegantly implement it

    boundary = chunk.chunk(_chunk_get_boundary, [vol], [bool], pad="extend")
    boundary = da.logical_and(boundary, vol)

    return boundary


def _get_expand_edt(vol, anisotropy):
    if not vol.flags["C_CONTIGUOUS"]:
        vol = np.ascontiguousarray(vol)

    if np.max(vol) == 0:
        return [np.zeros_like(vol, dtype=bool)]

    result = expand_parabola.expand_edt(
        vol,
        anisotropy=anisotropy,
        order="C",  # was C
        parallel=0,  # max CPU
    )

    assert np.all(np.isfinite(result))

    return [result < 0]


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
    vol,
    anisotropy,
    black_border,
    threshold,
    filter_idx=None,
):
    # computes euclidean distance transform (voxel-wise distance to nearest background)
    # vol: 3d volume (with 0 indicating back)
    # anisotropy: [z-size, y-size, x-size]
    # black_border: whether volume boundaries should be treated as foreground or background
    # threshold: edt distance threshold
    pad_width = [math.ceil(threshold / i) for i in anisotropy]
    if filter_idx is not None:
        vol = (vol != filter_idx).astype(vol.dtype)

    return chunk.chunk(
        _get_dt,
        [vol],
        [float],
        pad="extend",
        pad_width=pad_width,
        anisotropy=anisotropy,
        black_border=black_border,
    )


def get_expand_edt(
    vol,
    anisotropy,
    threshold,
):
    # computes euclidean distance transform (voxel-wise distance to nearest background)
    # vol: 3d volume (with 0 indicating back)
    # anisotropy: [z-size, y-size, x-size]
    # black_border: whether volume boundaries should be treated as foreground or background
    # threshold: edt distance threshold
    pad_width = [math.ceil(threshold / i) for i in anisotropy]

    return chunk.chunk(
        _get_expand_edt,
        [vol],
        [bool],
        pad="extend",
        pad_width=pad_width,
        anisotropy=anisotropy,
    )


def sphere_iteration(
    expanded,
    dt,
    vol,
    erode_delta,
    anisotropy,
):
    # performs sphere_iteration on volumes that may already be dilated
    # dt: edt from background in original volume
    # vol: original volume
    # erode_delta: nm, how much to dilate beyond edt
    # anisotropy: [z-size, y-size, x-size]
    # returns newly dilated volume

    edt = get_expand_edt(expanded * (dt + erode_delta), anisotropy, erode_delta)
    # logical_and
    result = da.logical_and(edt, vol)

    return result


def extract(
    vol,
    anisotropy,
    connectivity,
    max_erode,
    erode_delta,
    num_iter,
):
    # gets volume segmentation
    # vol: binary 3d volume
    # max_erode: int/float, thresholding distance to cut off spines
    # erode_delta: extra dilation over max_erode
    # anisotropy: [z-size, y-size, x-size]
    # connectivity: read cc3d docs

    # NOTE: assumes volume is connected

    dt = get_dt(vol, anisotropy, black_border=False, threshold=max_erode + erode_delta)
    remaining = dt >= max_erode
    # ignoring voxel_counts
    expanded, _ = chunk.chunk_cc3d(remaining, connectivity, k=1)

    # TODO: assert that final segmentation is only composed of single CC
    for _ in range(num_iter):
        expanded = sphere_iteration(expanded, dt, vol, erode_delta, anisotropy)

    others = da.logical_xor(vol, expanded)
    # label trunk as 1, others as 2
    merged = expanded + (2 * others)
    # segment everything
    # this assumes that the expanded trunk is larger than any of the non-eroded spines
    seg, voxel_counts = chunk.chunk_cc3d(
        merged,
        connectivity,
        k=False,
    )

    return seg, voxel_counts


@dask.delayed
def generate_seg_bbox(bboxes, voxel_counts):
    return np.concatenate((bboxes, voxel_counts[1:].reshape(-1, 1)), axis=1)


def main(base_path, id):
    if os.path.exists(os.path.join(base_path, "baseline", f"{str(id)}.h5")):
        return

    input = h5py.File(os.path.join(base_path, f"{str(id)}.h5")).get("main")
    input = dask_read_array(input)
    output = os.path.join(base_path, f"seg_{str(id)}.h5")

    seg, voxel_counts = extract(
        input, ANISOTROPY, CONNECTIVITY, MAX_ERODE, ERODE_DELTA, NUM_ITER
    )

    bboxes = chunk.chunk_bbox(seg)
    # cat voxel_counts to end of bboxes, removing background voxel countvc
    seg_bbox = generate_seg_bbox(bboxes, voxel_counts)
    seg_bbox = da.from_delayed(seg_bbox, shape=(np.nan, 8), dtype=UINT_DTYPE)

    file = dask_write_array(output, "seg", seg)
    file.create_dataset("seg_bbox", data=seg_bbox.compute())
    file.close()
