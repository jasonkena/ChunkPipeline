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
from chunk_sphere import get_dt, get_boundary, _chunk_get_boundary, _get_dt
from point import chunk_func_spine


def generate_simplex_noise(shape, feature_scale):
    try:
        import numba
    except ImportError:
        print("Please install numba to accelerate simplex_noise generation")
    idx = [np.linspace(0, 1 * feature_scale, i) for i in shape[::-1]]
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
        pad_width = (1, 2, 3)
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
                [f.get("output")],
                [f.get("input1"), f.get("input2")],
                chunk_size,
                lambda x, y: [x > y],
                num_workers,
                pad=pad,
                pad_width=pad_width,
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
        k = 13

        input = np.random.rand(*shape) > 0.8

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
            k,
        )

        # largest_k instead of connected_components, because of ordering by voxel_count
        gt = cc3d.largest_k(input, k=k, connectivity=connectivity)
        statistics = cc3d.statistics(gt)["voxel_counts"]
        statistics = np.concatenate([[statistics[0]], np.sort(statistics[1:])[::-1]])

        # cannot evaluate whether cc3d is equal because ordering cannot be guaranteed
        self.assertTrue(np.array_equal(output[1], statistics))
        # check whether k filtering causes incorrect results
        self.assertTrue(
            np.array_equal(
                output[1], cc3d.statistics(output[0][:].astype(np.uint))["voxel_counts"]
            )
        )

    def test_dt_sanity(self):
        anisotropy = (1, 2, 3)
        pad = 5
        input = np.ones((pad * 2 + 1, pad * 2 + 1, pad * 2 + 1))
        input[pad, pad, pad] = 0

        output = _get_dt(input, anisotropy, False)[0]

        assert output[0, pad, pad] == anisotropy[0] * pad
        assert output[pad, 0, pad] == anisotropy[1] * pad
        assert output[pad, pad, 0] == anisotropy[2] * pad

    def test_dt(self):
        num_workers = 2
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        anisotropy = (1, 2, 3)
        threshold = 40

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
        output_idx = output[:] <= threshold

        gt = _get_dt(f.get("input")[:], anisotropy, False)[0]
        gt_idx = gt <= threshold

        num_errors = np.logical_xor(output_idx, gt_idx)
        print(f"Errors: {np.sum(num_errors)}")

        self.assertTrue(np.array_equal(output[:][output_idx], gt[gt_idx]))
        self.assertTrue(np.array_equal(output_idx, gt_idx))

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
        num_workers = 2

        input = np.zeros(shape, dtype=int)
        input[40:60, 40:60, 40:60] = np.random.randint(0, 10, (20, 20, 20))

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)

        # get first row
        bbox = chunk.chunk_bbox(f.get("input"), chunk_size, num_workers)[0]
        assert bbox[0] == 1

        f.create_dataset(
            "seg",
            shape=(bbox[2] - bbox[1] + 1, bbox[4] - bbox[3] + 1, bbox[6] - bbox[5] + 1),
            dtype="u1",
        )
        seg = chunk.get_seg(
            f.get("seg"), f.get("input"), bbox, chunk_size, True, num_workers
        )

        output = chunk.chunk_argwhere(
            [seg],
            chunk_size,
            lambda params, vol: [vol, None],
            False,
            num_workers,
        )
        output += np.array([bbox[1], bbox[3], bbox[5]])
        output = output[np.lexsort(output.T)]

        idx = np.argwhere(input == bbox[0])
        idx = idx[np.lexsort(idx.T)]

        self.assertTrue(np.array_equal(output[:], idx))

    def test_argwhere_spine(self):
        # test both argwhere_seg and simple_chunk's bbox
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 2

        all = np.zeros(shape, dtype=int)
        spine = np.random.randint(0, 2, shape, dtype=int)
        all[40:60, 40:60, 40:60] = np.random.randint(0, 10, (20, 20, 20))

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("all", data=all)
        f.create_dataset("spine", data=spine)

        # get first row
        bbox = chunk.chunk_bbox(f.get("all"), chunk_size, num_workers)[0]
        assert bbox[0] == 1

        new_all = f.create_dataset(
            "new_all",
            shape=(bbox[2] - bbox[1] + 1, bbox[4] - bbox[3] + 1, bbox[6] - bbox[5] + 1),
            dtype="u1",
        )
        new_spine = f.create_dataset(
            "new_spine",
            shape=(bbox[2] - bbox[1] + 1, bbox[4] - bbox[3] + 1, bbox[6] - bbox[5] + 1),
            dtype="int",
        )

        new_all = chunk.get_seg(
            new_all, f.get("all"), bbox, chunk_size, True, num_workers
        )
        new_spine = chunk.get_seg(
            new_spine, f.get("spine"), bbox, chunk_size, True, num_workers
        )
        output = chunk.chunk_argwhere(
            [new_all, new_spine],
            chunk_size,
            lambda params, all, spine: chunk_func_spine(params, all, spine),
            "extend",
            num_workers,
        )
        output[:, :3] += np.array([bbox[1], bbox[3], bbox[5]])
        output = output[np.lexsort(output.T)]

        gt = _chunk_get_boundary(all == bbox[0])[0]
        gt = np.concatenate(
            [np.argwhere(gt), (spine == bbox[0])[gt].reshape(-1, 1)], axis=1
        )

        idx = gt[np.lexsort(gt.T)]

        self.assertTrue(np.array_equal(output, idx))

    def test_chunk_unique(self):
        # test both argwhere_seg and simple_chunk's bbox
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 2

        input = np.zeros(shape, dtype=int)
        input[40:60, 40:60, 40:60] = np.random.randint(0, 100, (20, 20, 20))

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)
        f.create_dataset("inverse", shape, dtype="uint16")

        gt_unique, gt_inverse = np.unique(input, return_inverse=True)
        unique, inverse = chunk.chunk_unique(
            f.get("input"),
            chunk_size,
            f.get("inverse"),
            num_workers,
        )
        self.assertTrue(np.array_equal(gt_unique, unique))
        self.assertTrue(np.array_equal(gt_inverse.reshape(shape), inverse[:]))

        unique = chunk.chunk_unique(
            f.get("input"),
            chunk_size,
            None,
            num_workers,
        )
        self.assertTrue(np.array_equal(gt_unique, unique))

    def test_get_seg(self):
        # test both argwhere_seg and simple_chunk's bbox
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        num_workers = 2

        input = np.zeros(shape, dtype=int)
        input[40:60, 40:60, 40:60] = np.random.randint(0, 10, (20, 20, 20))

        f = h5py.File("test.hdf5", "w")
        f.create_dataset("input", data=input)

        # get first row
        bbox = chunk.chunk_bbox(f.get("input"), chunk_size, num_workers)[0]
        assert bbox[0] == 1

        f.create_dataset(
            "output",
            shape=(bbox[2] - bbox[1] + 1, bbox[4] - bbox[3] + 1, bbox[6] - bbox[5] + 1),
            dtype="u1",
        )
        seg = chunk.get_seg(
            f.get("output"), f.get("input"), bbox, chunk_size, True, num_workers
        )

        gt = (
            input[bbox[1] : bbox[2] + 1, bbox[3] : bbox[4] + 1, bbox[5] : bbox[6] + 1]
            == bbox[0]
        )

        self.assertTrue(np.array_equal(seg[:], gt))

        # test merge_seg
        chunk.merge_seg(
            f.create_dataset(
                "reverse_output",
                shape=shape,
                dtype="uint16",
            ),
            f.create_dataset(
                "reverse_input",
                data=input[
                    bbox[1] : bbox[2] + 1, bbox[3] : bbox[4] + 1, bbox[5] : bbox[6] + 1
                ],
            ),
            bbox,
            chunk_size,
            lambda a, b: a + b,
            num_workers,
        )

        self.assertTrue(np.array_equal(input, f.get("reverse_output")[:]))


if __name__ == "__main__":
    unittest.main()
