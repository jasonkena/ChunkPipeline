import math
import numpy as np

import dask
import dask.array as da

import chunk_pipeline.tasks.chunk as chunk
import chunk_pipeline.tasks.sphere as sphere
from chunk_pipeline.tasks.sphere import get_boundary


def _chunk_zyx_idx(vol, uint_dtype, block_info):
    # compute zyx_idx in a chunkwise manner since da.arange broadcasting leads to memory errors on rechunking
    location = block_info[0]["array-location"]
    location = [x[0] for x in location]
    # ij for zyx ordering
    # offset by block coordinates
    z, y, x = np.meshgrid(
        *[
            np.arange(location[i], location[i] + vol.shape[i], dtype=uint_dtype)
            for i in range(3)
        ],
        indexing="ij",
        copy=False
    )
    return [z, y, x]


def chunk_zyx_idx(shape, row, chunk_size, uint_dtype):
    # NOTE: verify that row fits in uint_dtype
    # prevent casting to int
    row = row.astype(uint_dtype)
    temp = da.zeros(
        shape, chunks=chunk_size, dtype=bool
    )  # used only to compute block_info
    z, y, x = chunk.chunk(
        _chunk_zyx_idx,
        [temp],
        output_dataset_dtypes=[uint_dtype, uint_dtype, uint_dtype],
        uint_dtype=uint_dtype,
    )
    return z + row[1], y + row[3], x + row[5]


def chunk_idx(mask, array):
    return array[mask]


def chunk_mask(mask, arrays, row, chunk_size, uint_dtype):
    mask = mask.rechunk(chunk_size)
    idx = chunk_zyx_idx(mask.shape, row, chunk_size, uint_dtype)
    # splitting idx into 3-3D arrays so that binary indexing works
    arrays = list(idx) + arrays
    arrays = [array.rechunk(chunk_size) for array in arrays]

    results = [chunk_idx(mask, x) for x in arrays]
    # results = [x[mask] for x in arrays]
    idx, results = results[:3], results[3:]
    idx = da.stack(idx, axis=-1, allow_unknown_chunksizes=True)
    return idx, results


# def chunk_mask(mask, arrays, row, chunk_size):
#     mask = mask.rechunk(chunk_size)
#     mask_chunks = mask.blocks.ravel()
#
#     idx = chunk_idx(mask.shape, row)
#     arrays = [idx[..., i] for i in range(3)] + arrays
#     arrays = [arrays.rechunk(chunk_size) for arrays in arrays]
#
#     results = []
#     for array in arrays:
#         assert mask.chunks == array.chunks
#         # not sure how to rewrite with map_blocks/chunk.chunk
#         array_chunks = array.blocks.ravel()
#
#         results.append(
#             da.concatenate(
#                 [array_chunks[i][mask_chunks[i]] for i in range(len(mask_chunks))],
#                 axis=0,
#             )
#         )
#     # idx, rest of the arrays
#     __import__("pdb").set_trace()
#     # NOTE: idx and not arrays is what's causing the bottleneck
#     return results[0], results[1:]
#


@dask.delayed
def get_seed(skel, longest_path, row):
    radius = skel.radius  # real units

    vertices = skel.vertices
    vertices = vertices - np.array([row[1], row[3], row[5]]).reshape(1, -1)

    return vertices[longest_path].astype(int), radius[longest_path]


def task_generate_point_cloud(cfg, extracted):  # , skel):
    general = cfg["GENERAL"]
    # pc = cfg["PC"]
    row = extracted["row"]
    raw = extracted["raw"]
    spine = extracted["spine"]
    seg = extracted["seg"]

    # boundary = get_boundary(raw)
    # these are delayed objects
    # longest_path = skel["longest_path"]
    # skel = skel["skeleton"]

    # compute seed in extracted_seg coordinate system (as opposed to dataset coordinate system)
    # temp = get_seed(skel, longest_path, row)
    # seed, radius = temp[0], temp[1]  # since delayed objects cannot be unpacked
    # seed = da.from_delayed(seed, shape=(np.nan, 3), dtype=int)
    # radius = (
    #     da.from_delayed(radius, shape=(np.nan,), dtype=float) + pc["TRUNK_RADIUS_DELTA"]
    # )
    #
    # seeded = chunk.naive_chunk_seed(
    #     raw.shape, seed, radius, general["CHUNK_SIZE"], float
    # )
    # expanded = sphere.get_expand_edt(seeded, general["ANISOTROPY"], da.max(radius))

    idx, arrays = chunk_mask(
        raw,
        [spine, seg],
        row,
        general["CHUNK_SIZE"],
        general["UINT_DTYPE"],
    )

    result = {}
    result["idx"] = idx
    result["spine"] = arrays[0]
    result["seg"] = arrays[1]
    # result["boundary"] = arrays[1]
    # result["expanded"] = arrays[2]

    return result
