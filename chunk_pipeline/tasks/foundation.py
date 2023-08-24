import numpy as np
import cv2
import scipy.ndimage as nd
import cc3d

import dask
import dask.array as da

import chunk_pipeline.tasks.chunk as chunk
from chunk_pipeline.tasks.coarse import fill_and_remove_dust

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


def _normalize_empty(img, ignore_below_std):
    img = img.copy()
    valid = (img > 0) & (img < 255)
    # if blank slice
    if np.sum(valid) == 0:
        return [img]
    mean = img[valid].mean()
    std = img[valid].std()
    valid = valid & (img > mean + std * ignore_below_std)
    mean = img[valid].mean()
    img[~valid] = mean
    return [img]


def chunk_normalize_empty(vol, ignore_below_std):
    return chunk.chunk(
        _normalize_empty,
        [vol],
        output_dataset_dtypes=[vol.dtype],
        ignore_below_std=ignore_below_std,
    )


def _chunk_threshold_z_score(vol, z_score):
    return vol > (vol.mean() + z_score * vol.std())


def chunk_threshold_z_score(vol, z_score):
    return da.map_blocks(_chunk_threshold_z_score, vol, dtype=bool, z_score=z_score)


def task_foundation_seg(cfg, h5):
    general = cfg["GENERAL"]
    foundation = cfg["FOUNDATION"]
    vol = h5["main"]
    print("if np.concatenate is raised by cc3d, it means that there are 0 segments")

    # per-slice algorithms
    vol = da.rechunk(vol, chunks=foundation["CHUNK_SIZE"])
    normalized = chunk_normalize_empty(vol, foundation["IGNORE_BELOW_STD"])
    blurred = chunk_gaussian_blur(normalized, sigma=foundation["GAUSSIAN_SIGMA"])
    thresholded = chunk_threshold_z_score(
        blurred, z_score=foundation["THRESHOLD_Z_SCORE"]
    )
    filtered = fill_and_remove_dust(
        thresholded, foundation["DUST_THRESHOLD"], foundation["DUST_CONNECTIVITY"]
    )

    seg, voxel_counts = chunk.chunk_cc3d(
        filtered,
        connectivity=foundation["CONNECTIVITY"],
        k=False,
        uint_dtype=general["UINT_DTYPE"],
    )
    seg = da.rechunk(seg, chunks=general["CHUNK_SIZE"])

    return {"seg": seg, "voxel_counts": voxel_counts}
