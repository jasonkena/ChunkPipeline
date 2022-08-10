from settings import *
import numpy as np

chunk_location = [1, 2, 3]
zyx_idx_mask = np.zeros(list(CHUNK_SIZE) + [3], dtype=int)

zyx_idx_mask = zyx_idx_mask + np.array(chunk_location).reshape(1, 1, 1, 3)
