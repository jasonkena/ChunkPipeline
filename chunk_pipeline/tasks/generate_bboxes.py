import numpy as np
import h5py
import chunk
import sys
import os
from settings import *
from utils import dask_read_array
from dask.diagnostics import ProgressBar

with ProgressBar():
    main = h5py.File(os.path.join(sys.argv[1], "raw.h5")).get("main")
    main = dask_read_array(main)
    bbox = chunk.chunk_bbox(main).compute()
    np.save(os.path.join(sys.argv[1], "bbox.npy"), bbox)
