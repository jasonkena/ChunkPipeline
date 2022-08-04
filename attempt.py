import numpy as np
import cv2
import scipy.ndimage as nd
import skimage.morphology as morphology
import cc3d

import h5py
import glob
import os
import matplotlib.pyplot as plt

import dask
import dask.array as da
import dask.bag as db
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from tqdm import tqdm

from settings import *
import chunk
from utils import dask_write_array

from dask_memusage import install


def imshow(image):
    plt.figure()
    plt.imshow(image)


def read_png(file):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if image is not None:
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


def _chunk_grey_erosion(vol, structure, block_info=None):
    return [nd.grey_erosion(vol, structure=structure)]


def chunk_grey_erosion(vol, structure):
    assert [i % 2 == 1 for i in structure]
    pad_width = [(x - 1) // 2 for x in structure]

    return chunk.chunk(
        _chunk_grey_erosion,
        [vol],
        output_dataset_dtypes=[np.uint8],
        pad="extend",
        pad_width=tuple(pad_width),
        structure=structure,
    )


def _fill_and_remove_dust(vol):
    # big hack, since rechunking is expensive and list comprehensions are slow
    vol = nd.binary_fill_holes(vol)
    vol = cc3d.dust(vol, threshold=100, connectivity=26)
    return vol

def partial_fake_slice(func):
    # because rechunking is very expensive
    def inner_partial_fake_slice(vol):
        shape = vol.shape
        # N, H+1, W
        vol = np.pad(vol, [(0, 0), (0, 1), (0, 0)], mode="constant")
        vol = list(vol)
        # N*(H+1), W
        vol = np.concatenate(vol, axis=0)
        vol = func(vol)
        # N, H+1, W
        vol = np.stack(np.split(vol, shape[0], axis=0), axis=0)
        # N, H, W
        vol = vol[:, :-1, :]

        return vol
    return inner_partial_fake_slice

    # # big hack, since rechunking is expensive and list comprehensions are slow
    # result = np.zeros([vol.shape[0]*2, vol.shape[1], vol.shape[2]], dtype=vol.dtype)
    # result[::2, :, :] = vol
# def partial_fake_slice(func):
#     # because rechunking is very expensive
#     def inner_partial_fake_slice(vol):
#         return np.stack([func(x) for x in vol], axis=0)
#     return inner_partial_fake_slice

def fill_and_remove_dust(vol):
    return vol.map_blocks(partial_fake_slice(_fill_and_remove_dust), dtype=bool, token="fill_and_remove_dust")


def main():
    h5 = h5py.File(FILE, "w")
    vol = da.from_array(chunk.object_array(files), chunks=1)
    vol = da.map_blocks(
        lambda x: chunk.object_array([read_png(x[0])]), vol, dtype=object, token="read_png"
    )
    is_valid = vol.map_blocks(lambda x: x[0] is not None, dtype=bool, token="is_valid").compute()

    idx = fill_blanks(is_valid)
    vol = vol[idx]
    vol = da.map_blocks(
        lambda x: x[0][np.newaxis],
        vol,
        dtype=np.uint8,
        new_axis=[1, 2],
        chunks=[1] + list(SHAPE),
        token="new_axis",
    )

    # [N, H, W]
    # vol = dask.compute(*vol)
    original = vol
    vol = da.rechunk(vol, chunks=CHUNK_SIZE)
    eroded = chunk_grey_erosion(vol, np.ones([5, 1, 1]))

    thresholded = eroded > (da.mean(eroded) + da.std(eroded))
    filtered = fill_and_remove_dust(thresholded)

    cc, voxel_counts = chunk.chunk_cc3d(filtered, CONNECTIVITY, False, dtype=np.uint16)

    original, cc, voxel_counts = dask.compute(original, cc, voxel_counts)
    h5.create_dataset(
        "original", data=original, chunks=(128, 1600, 1600), compression="gzip"
    )
    h5.create_dataset("seg", data=cc, chunks=(128, 1600, 1600), compression="gzip")
    h5.create_dataset("voxel_counts", data=voxel_counts)
    print("done")



FILE = f"/mmfs1/data/adhinart/dendrite/r0.h5"
CHUNK_SIZE = (1600, 1600, 128)

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join("R0", "im_64nm", "*.png")))
    SHAPE = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).shape
    cluster = LocalCluster(
        n_workers=6, memory_limit="20GB", local_directory=SLURM_LOCAL_DIRECTORY
    )
    install(cluster.scheduler, "/mmfs1/data/adhinart/dendrite/memusage.csv")
    client = Client(cluster)
    print(cluster.dashboard_link)
    main()
