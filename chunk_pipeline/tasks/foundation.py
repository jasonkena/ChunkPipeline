import numpy as np
import cv2
import scipy.ndimage as nd
import cc3d

import dask
import dask.array as da

import chunk_pipeline.tasks.chunk as chunk

# NOTE: could rewrite this to use dask image, instead of reimplementing


def _otsu(img):
    assert img.shape[0] == 1
    img = img[0]
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return binary
    return [(th != 0)[np.newaxis]]


def chunk_otsu(vol):
    return chunk.chunk(_otsu, [vol], output_dataset_dtypes=[bool])


def _gaussian_blur(img, sigma):
    return [nd.gaussian_filter(img, sigma)]


def chunk_gaussian_blur(vol, sigma):
    return chunk.chunk(
        _gaussian_blur, [vol], output_dataset_dtypes=[float], sigma=sigma
    )


def _normalize_empty(img):
    img = img.copy()
    valid = (img > 0) & (img < 255)
    # if blank slice
    if np.sum(valid) == 0:
        return [img]
    mean = img[valid].mean()
    img[~valid] = mean
    return [img]


def chunk_normalize_empty(vol):
    return chunk.chunk(_normalize_empty, [vol], output_dataset_dtypes=[vol.dtype])


def task_foundation_seg(cfg, h5):
    general = cfg["GENERAL"]
    foundation = cfg["FOUNDATION"]
    vol = h5["main"]

    # per-slice algorithms
    vol = da.rechunk(vol, chunks=(1, -1, -1))
    normalized = chunk_normalize_empty(vol)
    blurred = chunk_gaussian_blur(normalized, sigma=foundation["GAUSSIAN_SIGMA"])
    otsu = chunk_otsu(blurred)
    otsu = da.rechunk(otsu, chunks=general["CHUNK_SIZE"])

    seg, voxel_counts = chunk.chunk_cc3d(
        otsu,
        connectivity=foundation["CONNECTIVITY"],
        k=False,
        uint_dtype=general["UINT_DTYPE"],
    )

    return {"seg": seg, "voxel_counts": voxel_counts}
