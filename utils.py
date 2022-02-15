import numpy as np
import torch


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
