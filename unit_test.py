import unittest
import numpy as np
import h5py
import chunk
import cc3d
import os

from imu.io import get_bb_all3d


class ChunkTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        try:
            os.remove("test.hdf5")
        except:
            pass

    def test_simple_chunk_output(self):
        return
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input1 = np.random.rand(*shape)
        input2 = np.random.rand(*shape)
        # cannot test function which depends on dtype (i.e., floats)
        gt = input1 > input2

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input1", shape, dtype="f")[:] = input1
        f.create_dataset("input2", shape, dtype="f")[:] = input2
        f.create_dataset("output", shape, dtype="f"),

        for pad in ["zero", "extend", "half_extend", False]:
            output = chunk.simple_chunk(
                f.get("output"),
                [f.get("input1"), f.get("input2")],
                chunk_size,
                pad,
                False,
                lambda x, y: x > y,
            )
            print(f"pad method: {pad}")
            self.assertTrue(np.all(output[:] == gt))

    def test_chunk_bbox(self):
        return
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input = np.random.randint(0, 10, shape)
        gt = get_bb_all3d(input)

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", shape, dtype="i")[:] = input

        output = chunk.chunk_bbox(f.get("input"), chunk_size)
        self.assertTrue(np.all(output == gt))

    def test_cc3d(self):
        shape = (2, 2, 1)
        chunk_size = (1, 1, 1)
        # shape = (100, 100, 100)
        # chunk_size = (9, 8, 7)
        connectivity = 26

        input = np.zeros(shape)
        input[1, 0] = 1
        input[0, 1] = 1
        # input = np.random.rand(*shape) > 0.5
        # input = input[:2]

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", shape, dtype="u1")[:] = input
        f.create_dataset("output", shape, dtype="i")

        output = chunk.chunk_cc3d(
            f.get("output"), f.get("input"), chunk_size, connectivity=connectivity
        )
        N = output[1].shape[0] - 1

        # largest_k instead of connected_components, because of ordering by voxel_count
        gt, gt_N = cc3d.largest_k(input, k=N, return_N=True, connectivity=connectivity)
        n_gt, n_gt_N = cc3d.connected_components(
            input, return_N=True, connectivity=connectivity
        )
        print(f"N: {N}, gt_N: {gt_N}")

        __import__("pdb").set_trace()
        self.assertTrue(N == gt_N)
        self.assertTrue(np.all(output[0][:] == gt))


if __name__ == "__main__":
    unittest.main()
