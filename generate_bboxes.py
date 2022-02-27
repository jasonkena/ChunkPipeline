import numpy as np
import h5py
import chunk
from settings import *

file = h5py.File("seg_den_6nm.h5")
bbox = chunk.chunk_bbox(file.get("main"), CHUNK_SIZE, NUM_WORKERS)
np.save("bbox.npy", bbox)
