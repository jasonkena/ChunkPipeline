import glob
import zarr
import numpy as np

paths = [
    "/mmfs1/data/adhinart/dendrite/data/seg_den",
    "/mmfs1/data/adhinart/dendrite/data/human",
    "/mmfs1/data/adhinart/dendrite/data/mouse",
]

anisotropies = [(30, 6, 6), (30, 8, 8), (30, 8, 8)]


def get_average_skel_length(path, anisotropy):
    dirs = sorted(glob.glob(path + "/skeleton_*"))
    anisotropy = np.array(anisotropy, dtype=float)
    lengths = []
    for dir in dirs:
        group = zarr.open_group(dir)
        skel = group["_attrs"][0]
        skel, longest_path = skel["skeleton"], skel["longest_path"]
        longest_path = (skel.vertices * anisotropy)[longest_path]

        l2 = np.sqrt(np.sum((longest_path[1:] - longest_path[:-1]) ** 2, axis=1))
        total_length = np.sum(l2)
        lengths.append(total_length)
    print(path)
    print(f"mean: {np.mean(lengths)}")
    print(f"std: {np.std(lengths)}")
    print(f"min: {np.min(lengths)}")
    print(f"max: {np.max(lengths)}")


if __name__ == "__main__":
    for path, anisotropy in zip(paths, anisotropies):
        get_average_skel_length(path, anisotropy)
