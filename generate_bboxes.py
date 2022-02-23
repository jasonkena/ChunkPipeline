import numpy as np
import h5py
import chunk
from settings import *

file = h5py.File("./den_ruilin_v2_16nm.h5")
bbox = chunk.chunk_bbox(file.get("main"), CHUNK_SIZE)
np.save("bbox.npy", bbox)
