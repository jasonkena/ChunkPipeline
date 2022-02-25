import h5py
import numpy as np
import math
from chunk import get_is_first_unique

ANISOTROPY = (30, 6, 6)
RESOLUTION = 200

# Tool to guesstimate required chunk_size to cover a certain number of points


def binning(point_cloud, resolution, anisotropy):
    point_cloud = point_cloud.astype(int)
    chunk_size = np.array([math.ceil(resolution / anisotropy[i]) for i in range(3)])
    point_cloud[:, :3] = np.floor(point_cloud[:, :3] / chunk_size)
    unique, counts = np.unique(point_cloud[:, :3], axis=0, return_counts=True)

    max_range = (unique.max(axis=0) - unique.min(axis=0)).max()
    double_distances = np.max(
        np.absolute(unique.reshape(1, -1, 3) - unique.reshape(-1, 1, 3)), axis=-1
    ).reshape(-1)
    double_counts = np.repeat(counts.reshape(1, -1), counts.shape[0], axis=0).reshape(
        -1
    )

    order = np.argsort(double_distances)
    double_distances = double_distances[order]
    double_counts = double_counts[order]

    # +1 to satisfy !=0 of is_first unique
    is_first_unique = np.argwhere(get_is_first_unique(double_distances + 1))[:, 0]
    # sphere radius, not chunk_size
    final_distances = double_distances[is_first_unique] * resolution + resolution / 2
    is_first_unique = np.concatenate([is_first_unique, [double_distances.shape[0]]])

    final_counts = []
    for i in range(is_first_unique.shape[0] - 1):
        final_counts.append(
            np.sum(double_counts[is_first_unique[i] : is_first_unique[i + 1]])
        )
    final_counts = np.array(final_counts) / unique.shape[0]
    final_counts = np.cumsum(final_counts)

    return final_distances, final_counts


if __name__ == "__main__":
    max_shape = h5py.File("seg_den_6nm.h5").get("main").shape
    point_cloud_1 = np.load("results/2.npy")
    final_distances, final_counts = binning(point_cloud_1, RESOLUTION, ANISOTROPY)
    print(f"Chunk radius: {final_distances}")
    print(final_counts)
