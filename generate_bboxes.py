import numpy as np
import h5py
import chunk
import sys
import os
from settings import *

file = h5py.File(os.path.join(sys.argv[1], "raw.h5"))
bbox = chunk.chunk_bbox(file.get("main"), CHUNK_SIZE, NUM_WORKERS)
np.save(os.path.join(sys.argv[1], "bbox.npy"), bbox)
