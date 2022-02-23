import unittest
import numpy as np
import h5py
import chunk
import cc3d
import os

from imu.io import get_bb_all3d
import numba
import opensimplex
import edt
from chunk_sphere import get_dt


def generate_simplex_noise(shape, feature_scale):
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

        input = np.random.rand(*shape) > 0.8

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)
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
        statistics = np.sort(cc3d.statistics(gt)["voxel_counts"])[::-1]
        print(f"N: {N}, gt_N: {gt_N}")

        self.assertTrue(N == gt_N)
        # cannot evaluate whether cc3d is equal because ordering cannot be guaranteed
        self.assertTrue(np.array_equal(output[1], statistics))

    def test_dt(self):
        shape = (100, 100, 100)
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


if __name__ == "__main__":
    unittest.main()
