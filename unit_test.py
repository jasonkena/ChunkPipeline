import unittest
import numpy as np
import h5py
import chunk
import cc3d
import os
import torch
import torch.nn.functional as F
from utils import pad_vol

from imu.io import get_bb_all3d
import opensimplex
import edt
from chunk_sphere import get_dt, get_boundary
from point import chunk_argwhere


def generate_simplex_noise(shape, feature_scale):
    try:
        import numba
    except ImportError:
        print("Please install numba to accelerate simplex_noise generation")
    idx = [np.linspace(0, 1 * feature_scale, i) for i in shape]
    return opensimplex.noise3array(*idx)


class ChunkTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        opensimplex.seed(0)
        try:
            os.remove("test.hdf5")
        except:
            pass

    def test_simple_chunk_output(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 2

        input1 = np.random.rand(*shape)
        input2 = np.random.rand(*shape)
        # cannot test function which depends on dtype (i.e., floats)
        gt = input1 > input2

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input1", data=input1)
        f.create_dataset("input2", data=input2)
        f.create_dataset("output", shape, dtype="f"),

        for pad in ["zero", "extend", "half_extend", False]:
            output = chunk.simple_chunk(
                f.get("output"),
                [f.get("input1"), f.get("input2")],
                chunk_size,
                lambda x, y: x > y,
                num_workers,
                pad=pad,
            )
            print(f"pad method: {pad}")
            self.assertTrue(np.array_equal(output[:], gt))

    def test_chunk_bbox(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 2

        input = np.random.randint(0, 10, shape)
        gt = get_bb_all3d(input)

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)

        output = chunk.chunk_bbox(f.get("input"), chunk_size, num_workers)
        self.assertTrue(np.array_equal(output, gt))

    def test_cc3d(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        connectivity = 26
        num_workers = 2

        input = np.random.rand(*shape) > 0.5

        f = h5py.File("test.hdf5", "w")
        # NOTE: the input has to be be u1 instead of b1 for some reason
        f.create_dataset("input", data=input, dtype="u1")
        f.create_dataset("output", shape, dtype="i")
        group_cache = f.create_group("cache")

        output = chunk.chunk_cc3d(
            f.get("output"),
            f.get("input"),
            group_cache,
            chunk_size,
            connectivity,
            num_workers,
        )
        N = output[1].shape[0] - 1

        # largest_k instead of connected_components, because of ordering by voxel_count
        gt, gt_N = cc3d.largest_k(input, k=N, return_N=True, connectivity=connectivity)
        statistics = cc3d.statistics(gt)["voxel_counts"]
        statistics = np.concatenate([[statistics[0]], np.sort(statistics[1:])[::-1]])
        print(f"N: {N}, gt_N: {gt_N}")

        self.assertTrue(N == gt_N)
        # cannot evaluate whether cc3d is equal because ordering cannot be guaranteed
        self.assertTrue(np.array_equal(output[1], statistics))

    def test_dt(self):
        shape = (10, 10, 10)
        num_workers = 2
        # NOTE: NOTE: NOTE: CHUNK SIZE NOT ACCESSED
        chunk_size = (1, 1, 1)
        # chunk_size = (9, 8, 7)
        anisotropy = (30, 6, 6)
        threshold = 0
        input = generate_simplex_noise(shape, 0.1)  # > 0 #np.random.rand(*shape) > 0.5
        input = input > input.mean()

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)
        f.create_dataset("output", shape, dtype="f")

        output = get_dt(
            f.get("output"),
            f.get("input"),
            chunk_size,
            anisotropy,
            False,
            threshold,
            num_workers,
        )

        # largest_k instead of connected_components, because of ordering by voxel_count
        gt = edt.edt(
            input,
            anisotropy=anisotropy[::-1],
            black_border=False,
            order="C"
            if input.flags["C_CONTIGUOUS"]
            else "F",  # depends if Fortran contiguous or not
            parallel=0,  # max CPU
        )

        self.assertTrue(np.array_equal(output[:], gt))

    def test_get_boundary(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 2

        input = np.random.rand(*shape) > 0.5

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)
        f.create_dataset("output", shape, dtype="i")

        output = get_boundary(
            f.get("output"),
            f.get("input"),
            chunk_size,
            num_workers,
        )

        padded_vol = torch.from_numpy(pad_vol(input, [3, 3, 3]))
        input = torch.from_numpy(input)
        boundary = torch.logical_and(
            F.max_pool3d((~padded_vol).float().unsqueeze(0), kernel_size=3, stride=1),
            input,
        ).squeeze(0)

        self.assertTrue(np.array_equal(output[:], boundary))

    def test_argwhere_seg(self):
        # test both argwhere_seg and simple_chunk's bbox
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 0

        input = np.zeros(shape, dtype=int)
        input[50:60, 50:60, 50:60] = np.random.randint(0, 10, (10, 10, 10))

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)

        # get first row
        bbox = chunk.chunk_bbox(f.get("input"), chunk_size, num_workers)[0]
        bbox[0] = 1

        output = chunk_argwhere(
            [f.get("input")],
            chunk_size,
            lambda vol: [vol==bbox[0], None],
            num_workers,
        )
        output = output[np.lexsort(output.T)]

        idx = np.argwhere(input == bbox[0])
        idx = idx[np.lexsort(idx.T)]

        self.assertTrue(np.array_equal(output[:], idx))

    # TODO: implement get_seg testing


if __name__ == "__main__":
    unittest.main()
