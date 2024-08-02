import numpy as np
from numba import njit
from cloudvolume import CloudVolume
from to_precomputed import get_chunks
from tqdm import tqdm

from typing import Tuple
from utils import get_conf

from joblib import Parallel, delayed

"""
Goal here is to find centroid of coordinates where spine touches trunk
"""


def offsets(connectivity: int):
    assert connectivity in [6, 18, 26], "Connectivity must be 6, 18, or 26"

    offsets = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if connectivity == 6 and (abs(i) + abs(j) + abs(k) == 1):
                    offsets.append((i, j, k))
                elif connectivity == 18 and (abs(i) + abs(j) + abs(k) in [1, 2]):
                    offsets.append((i, j, k))
                elif connectivity == 26 and (i, j, k) != (0, 0, 0):
                    offsets.append((i, j, k))

    return np.array(offsets)


@njit
def _find_mismatched_coords(
    arr: np.ndarray, offsets: np.ndarray, chunk_offset: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns sum coordinates, and the number of elems for that sum, taking into account chunk_offset

    Parameters
    ----------
    arr : 3D
    offsets : [Nx3]
    """
    coord_dtype = np.float64
    count_dtype = np.int64

    max_id = arr.max()  # Find the maximum ID value to set array sizes

    # Use arrays for coordinates and counts
    agg_coords = np.zeros((max_id + 1, max_id + 1, 3), dtype=coord_dtype)
    agg_counts = np.zeros((max_id + 1, max_id + 1), dtype=count_dtype)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if arr[i, j, k] == 0:
                    continue  # Skip if the current cell is zero
                for offset in offsets:
                    ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
                    if (
                        0 <= ni < arr.shape[0]
                        and 0 <= nj < arr.shape[1]
                        and 0 <= nk < arr.shape[2]
                    ):
                        if arr[ni, nj, nk] != 0 and arr[i, j, k] != arr[ni, nj, nk]:
                            id1 = arr[i, j, k]
                            id2 = arr[ni, nj, nk]
                            agg_coords[id1, id2] += np.array(
                                [
                                    i + offset[0] / 2 + chunk_offset[0],
                                    j + offset[1] / 2 + chunk_offset[1],
                                    k + offset[2] / 2 + chunk_offset[2],
                                ]
                            )
                            agg_counts[id1, id2] += 1

    return agg_coords, agg_counts


def find_mismatched_coords(
    vol: CloudVolume, chunk: Tuple[slice, slice, slice], connectivity: int
):
    assert len(vol.shape) == 4 and vol.shape[3] == 1, "Volume must be 3D"
    # [3D now]
    vol = vol[chunk].squeeze(-1)
    agg_coords, agg_counts = _find_mismatched_coords(
        vol, offsets(connectivity), tuple([s.start for s in chunk])
    )

    # assert symmetric matrices
    assert np.allclose(agg_coords, agg_coords.transpose(1, 0, 2))
    assert np.allclose(agg_counts, agg_counts.transpose(1, 0))

    return agg_coords, agg_counts


def main(conf):
    """
    outputs the mean coordinates of touching centroids [max_id+1, max_id+1, 3]
    """
    vol = CloudVolume(f"file://{conf.data.output_layer}")
    chunks = get_chunks(vol.shape, conf.chunk_size)

    res = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs_touching_centroids)(
                delayed(find_mismatched_coords)(
                    vol, chunk, conf.touching_centroids.connectivity
                )
                for chunk in chunks
            ),
            total=len(chunks),
            leave=False,
        )
    )

    max_shape = max([res[i][0].shape[0] for i in range(len(res))])
    agg_coords = np.zeros((max_shape, max_shape, 3), dtype=np.float64)
    agg_counts = np.zeros((max_shape, max_shape), dtype=np.int64)

    for coords, counts in res:
        agg_coords[: coords.shape[0], : coords.shape[1]] += coords
        agg_counts[: counts.shape[0], : counts.shape[1]] += counts

    assert np.allclose(agg_coords, agg_coords.transpose(1, 0, 2))
    assert np.allclose(agg_counts, agg_counts.transpose(1, 0))

    agg_coords[agg_counts > 0] /= agg_counts[agg_counts > 0][:, None]

    np.savez(conf.data.touching, mean_coords=agg_coords, counts=agg_counts)


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
