TASK = "human"

GENERAL__ANISOTROPY = (30, 8, 8)
base = "/mmfs1/data/adhinart/dendrite/raw/"
H5 = {
    "raw": (f"{base}{TASK}_raw.h5", "main"),
    "spine": (f"{base}{TASK}_spine.h5", "main"),
    "seg": (f"{base}{TASK}_seg.h5", "main"),
}
L1__NUM_SAMPLE = 50000  # GUI example has 40000 points
L1__NOISE_STD = 30.0  # determine this by eyeballing downsampled pointcloud
