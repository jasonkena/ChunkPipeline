import dask
import dask.array as da

import chunk_pipeline.tasks.chunk as chunk

def _chunk_overlap(x, depth):
    # get slices for neighbors
    results = []
    for dim in depth:
        s_left = [slice(None)] * x.ndim
        s_right = s_left.copy()

        s_left[dim] = slice(0, depth[dim][0])
        s_right[dim] = slice(x.shape[dim] - depth[dim][1], x.shape[dim])

        results.append(x[s_left])
        results.append(x[s_right])
    return results


def chunk_overlap(x, depth):
