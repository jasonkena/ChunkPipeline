import numpy as np
import h5py
from imu.io import get_bb_all3d

file = np.array(h5py.File("./den_ruilin_v2_16nm.h5").get("main")[:])
bbox = get_bb_all3d(file)
np.save("bbox.npy", bbox)
