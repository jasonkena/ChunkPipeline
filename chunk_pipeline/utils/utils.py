import uuid
import numpy as np
import dask
import dask.array as da
from distributed import get_client

import logging


def pad_vol(vol, kernel_shape):
    # given a 3d volume and a kernel shape, pad the input so that applying the kernel on the volume will result in a volume with the original shape
    # vol: 3d volume
    # kernel_shape: [z_size, y_size, x_size]
    assert np.all(np.array(kernel_shape) % 2 == 1)
    padded_vol = np.pad(
        vol,
        [
            [kernel_shape[0] // 2] * 2,
            [kernel_shape[1] // 2] * 2,
            [kernel_shape[2] // 2] * 2,
        ],
    )
    return padded_vol


def extend_bbox(bbox, max_shape):
    bbox = bbox.copy()
    bbox[1] = max(0, bbox[1] - 1)
    bbox[3] = max(0, bbox[3] - 1)
    bbox[5] = max(0, bbox[5] - 1)
    # -1 because of inclusive indexing
    bbox[2] = min(max_shape[0] - 1, bbox[2] + 1)
    bbox[4] = min(max_shape[1] - 1, bbox[4] + 1)
    bbox[6] = min(max_shape[2] - 1, bbox[6] + 1)

    return bbox


def object_array(input):
    # properly create an object array
    result = np.empty(len(input), dtype=object)
    result[:] = input
    return result


def publish(name, collection, persist=False):
    # if not persist:
    #     raise ValueError("non persisted collections currently broken")
    client = get_client()
    hash = uuid.uuid4().hex
    name = f"{name}-{hash}"

    if client.get_dataset(name, None) is not None:
        raise KeyError(f"Dataset {name} already exists")
    if persist:
        # worker = get_worker()
        # other_workers = [w for w in client.scheduler_info()["workers"] if w != worker.address]
        # persist to prevent vol from being garbage collected
        # persist it on the current worker to prevent data transfer
        # vol = client.persist(vol, workers=worker.address)
        # vol = client.persist(vol, workers=other_workers)
        collection = client.persist(collection)

    client.publish_dataset(collection, name=name)
    return name


def _normalize_dataset(_, delayed_name, block_info=None):
    client = get_client()
    # [(start0, stop0), (start1, stop1), ...]
    array_location = block_info[0]["array-location"]
    dataset = client.get_dataset(delayed_name)
    # create slices for each dimension
    slices = [slice(start, stop, 1) for start, stop in array_location]
    vol = dataset[tuple(slices)]

    logging.error(delayed_name)
    logging.error(slices)
    logging.error(vol)
    # return concrete values and not futures
    return client.compute(vol, sync=True)


def normalize_dataset(delayed_name, shape, dtype, chunk_size):
    result = da.map_blocks(
        _normalize_dataset,
        da.zeros(shape, dtype=dtype, chunks=chunk_size),
        dtype=dtype,
        delayed_name=delayed_name,
    )
    return result
