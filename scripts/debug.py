import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dask
import dask.array as da
import chunk_pipeline.tasks.point as point

# chunk_location = [1, 2, 3]
# zyx_idx_mask = np.zeros(list(CHUNK_SIZE) + [3], dtype=int)
#
# zyx_idx_mask = zyx_idx_mask + np.array(chunk_location).reshape(1, 1, 1, 3)

# point
location = [1, 2, 3]
shape = [512] * 3
dtype = int
# dtype = np.uint16

z, y, x = np.meshgrid(
    *[np.arange(location[i], location[i] + shape[i], dtype=dtype) for i in range(3)],
    indexing="ij",
    copy=False
)
z + 1
y + 1
x + 1
