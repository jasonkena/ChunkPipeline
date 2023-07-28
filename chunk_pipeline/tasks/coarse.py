import glob

import numpy as np
import cv2
import scipy.ndimage as nd
import cc3d

import dask
import dask.array as da

import chunk_pipeline.tasks.chunk as chunk
from chunk_pipeline.utils import object_array


def read_png(file, apply_clahe, clip_limit, tile_grid_size):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if image is not None and apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image = clahe.apply(image)
        # image = image.astype(float)
        # image = image / 255.0
    return image


def fill_blanks(is_valid):
    # fill in blank z vol with preceding valid image
    assert is_valid[0]
    valid_idx = np.nonzero(is_valid)[0]
    idx = valid_idx[np.cumsum(is_valid) - 1]
    return idx


def _chunk_grey_erosion(vol, structure):
    return [nd.grey_erosion(vol, structure=structure)]


def chunk_grey_erosion(vol, structure, uint_dtype):
    structure = np.array(structure)
    assert [i % 2 == 1 for i in structure.shape]
    pad_width = [(x - 1) // 2 for x in structure.shape]

    return chunk.chunk(
        _chunk_grey_erosion,
        [vol],
        output_dataset_dtypes=[uint_dtype],
        pad="extend",
        pad_width=tuple(pad_width),
        structure=structure,
    )


def _fill_and_remove_dust(vol, dust_threshold, dust_connectivity):
    vol = nd.binary_fill_holes(vol)
    vol = cc3d.dust(vol, threshold=dust_threshold, connectivity=dust_connectivity)
    return vol


def fill_and_remove_dust(vol, dust_threshold, dust_connectivity):
    vol = vol.map_blocks(
        _fill_and_remove_dust,
        dtype=bool,
        dust_threshold=dust_threshold,
        dust_connectivity=dust_connectivity,
    )
    return vol


def task_generate_original(cfg):
    general = cfg["GENERAL"]
    original = cfg["COARSE_ORIGINAL"]

    files = sorted(glob.glob(original["IMAGE_PATH"]))
    # since first file has irregular shape
    image = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    shape, uint_dtype = image.shape, image.dtype

    vol = da.from_array(object_array(files), chunks=1)
    vol = da.map_blocks(
        lambda x: object_array(
            [
                read_png(
                    x[0],
                    apply_clahe=original["APPLY_CLAHE"],
                    clip_limit=original["CLIP_LIMIT"],
                    tile_grid_size=original["TILE_GRID_SIZE"],
                )
            ]
        ),
        vol,
        dtype=object,
        name="read_png",
    )
    is_valid = vol.map_blocks(
        lambda x: x[0] is not None, dtype=bool, name="is_valid"
    ).compute()

    idx = fill_blanks(is_valid)
    # without rechunk, chunksize won't be one
    vol = da.rechunk(vol[idx], chunks=1)
    vol = da.map_blocks(
        lambda x: x[0][np.newaxis],
        vol,
        dtype=np.uint8,
        new_axis=[1, 2],
        chunks=[1] + list(shape),
        name="new_axis",
    )

    vol = da.rechunk(vol, chunks=general["CHUNK_SIZE"])
    return {"original": vol, "shape": "shape", "uint_dtype": uint_dtype}
    # generate stack of images


def task_coarse_segment(cfg, original):
    general = cfg["GENERAL"]
    coarse = cfg["COARSE"]

    original, shape, uint_dtype = (
        original["original"],
        original["shape"],
        original["uint_dtype"],
    )

    eroded = chunk_grey_erosion(original, coarse["EROSION_STRUCTURE"], uint_dtype)

    thresholded = eroded > (
        da.mean(eroded) + coarse["THRESHOLD_Z_SCORE"] * da.std(eroded)
    )
    filtered = fill_and_remove_dust(
        thresholded, coarse["DUST_THRESHOLD"], coarse["DUST_CONNECTIVITY"]
    )

    cc, voxel_counts = chunk.chunk_cc3d(
        filtered, coarse["CONNECTIVITY"], False, uint_dtype=general["UINT_DTYPE"]
    )

    return {"seg": cc, "voxel_counts": voxel_counts}
