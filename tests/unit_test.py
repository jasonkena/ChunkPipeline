import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import cc3d
import torch
import torch.nn.functional as F
from chunk_pipeline.utils import pad_vol
import math
from multiprocessing.pool import ThreadPool

from imu.io import get_bb_all3d
import opensimplex
import edt
import dask
import dask.array as da
from dask.diagnostics import ProgressBar

import chunk_pipeline.tasks.chunk as chunk
import chunk_pipeline.tasks.sphere as sphere

# import chunk_pipeline.tasks.inference as inference
# import chunk_pipeline.tasks.evaluation as evaluation

from stardist.data import test_image_nuclei_3d
from scipy.ndimage import rotate
from stardist.matching import matching as stardist_matching

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# from settings import *


def generate_simplex_noise(shape, feature_scale):
    try:
        import numba
    except ImportError:
        print("Please install numba to accelerate simplex_noise generation")
    idx = [np.linspace(0, 1 * feature_scale, i) for i in shape[::-1]]
    return opensimplex.noise3array(*idx)


class ChunkTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     cluster = SLURMCluster(
    #         job_name="dendrite_test",
    #         queue=SLURM_PARTITIONS,
    #         cores=20,
    #         memory="10GB",
    #     )
    #     cluster.scale(10)
    #     client = Client(cluster)
    #
    def setUp(self):
        np.random.seed(0)
        opensimplex.seed(0)
        try:
            os.remove("test.hdf5")
        except:
            pass
        # dask.config.set(pool=ThreadPool(20))
        # client = Client()
        # print(client.cluster)

    def test_chunk_output(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        pad_width = (1, 2, 3)

        input1 = np.random.rand(*shape)
        input2 = np.random.rand(*shape)
        # cannot test function which depends on dtype (i.e., floats)
        gt = input1 > input2

        input_datasets = [da.from_array(x, chunks=chunk_size) for x in [input1, input2]]

        def dumb(a, b, block_info=None):
            return [a > b]

        for pad in ["extend", "half_extend", False]:
            output = chunk.chunk(
                dumb,
                input_datasets,
                output_dataset_dtypes=["b"],
                pad=pad,
                pad_width=pad_width,
            )
            print(f"pad method: {pad}")
            # output.visualize(filename=f"{pad}.svg")
            with ProgressBar():
                self.assertTrue(np.array_equal(output.compute(), gt))

    def test_chunk_bbox(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input = np.random.randint(0, 2 ** 16 - 1, shape)
        gt = get_bb_all3d(input)

        output = chunk.chunk_bbox(da.from_array(input, chunks=chunk_size)).compute()
        self.assertTrue(np.array_equal(output, gt))

    def test_dask_cc3d(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        connectivity = 26
        k = 13

        input = np.random.rand(*shape) > 0.8

        output = chunk.chunk_cc3d(
            da.from_array(input, chunks=chunk_size),
            connectivity,
            k,
        )
        output = dask.compute(*output)

        # largest_k instead of connected_components, because of ordering by voxel_count
        gt = cc3d.largest_k(input, k=k, connectivity=connectivity)
        statistics = cc3d.statistics(gt)["voxel_counts"]
        statistics = np.concatenate([[statistics[0]], np.sort(statistics[1:])[::-1]])

        # cannot evaluate whether cc3d is equal because ordering cannot be guaranteed
        self.assertTrue(np.array_equal(output[1], statistics))
        # check whether k filtering causes incorrect results
        self.assertTrue(
            np.array_equal(
                output[1], cc3d.statistics(output[0].astype(np.uint))["voxel_counts"]
            )
        )

    def test_dt_sanity(self):
        anisotropy = (1, 2, 3)
        pad = 5
        input = np.ones((pad * 2 + 1, pad * 2 + 1, pad * 2 + 1))
        input[pad, pad, pad] = 0

        output = sphere._get_dt(input, anisotropy, False, None)[0]

        assert output[0, pad, pad] == anisotropy[0] * pad
        assert output[pad, 0, pad] == anisotropy[1] * pad
        assert output[pad, pad, 0] == anisotropy[2] * pad

    def test_dt(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)
        anisotropy = (1, 2, 3)
        threshold = 40

        input = generate_simplex_noise(shape, 0.1)  # > 0 #np.random.rand(*shape) > 0.5
        input = input > input.mean()

        output = sphere.get_dt(
            da.from_array(input, chunks=chunk_size),
            anisotropy,
            False,
            threshold,
        ).compute()
        output_idx = output[:] <= threshold

        gt = sphere._get_dt(input, anisotropy, False, None)[0]
        gt_idx = gt <= threshold

        num_errors = np.logical_xor(output_idx, gt_idx)
        print(f"Errors: {np.sum(num_errors)}")

        self.assertTrue(np.array_equal(output[:][output_idx], gt[gt_idx]))
        self.assertTrue(np.array_equal(output_idx, gt_idx))

    def test_dask_get_boundary(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input = np.random.rand(*shape) > 0.5

        with ProgressBar():
            output = sphere.get_boundary(da.from_array(input, chunks=chunk_size))
            output = output.compute()

        padded_vol = torch.from_numpy(pad_vol(input, [3, 3, 3]))
        input = torch.from_numpy(input)
        boundary = (
            torch.logical_and(
                F.max_pool3d(
                    (~padded_vol).float().unsqueeze(0), kernel_size=3, stride=1
                ),
                input,
            )
            .squeeze(0)
            .numpy()
        )

        self.assertTrue(np.array_equal(output, boundary))

    def test_chunk_nonzero(self):
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input = np.random.rand(*shape) > 0.5
        gt = np.stack(np.nonzero(input), axis=1)

        output = chunk.chunk_nonzero(da.from_array(input, chunks=chunk_size))
        self.assertTrue(np.array_equal(gt, output.compute()))

        extra = np.random.randint(0, 100, shape)
        gt_extra = np.concatenate([gt, extra[input].reshape(-1, 1)], axis=1)
        output_extra = chunk.chunk_nonzero(
            da.from_array(input, chunks=chunk_size),
            extra=da.from_array(extra, chunks=chunk_size),
        )
        self.assertTrue(np.array_equal(gt_extra, output_extra.compute()))

    def test_dask_get_seg(self):
        # test both argwhere_seg and simple_chunk's bbox
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input = np.zeros(shape, dtype=int)
        input[40:60, 40:60, 40:60] = np.random.randint(0, 10, (20, 20, 20))

        input_da = da.from_array(input, chunks=chunk_size)

        # get first row
        bbox = chunk.chunk_bbox(input_da).compute()[0]
        assert bbox[0] == 1

        seg = chunk.get_seg(input_da, bbox, filter_id=True).compute()

        gt = (
            input[bbox[1] : bbox[2] + 1, bbox[3] : bbox[4] + 1, bbox[5] : bbox[6] + 1]
            == bbox[0]
        )

        self.assertTrue(np.array_equal(seg, gt))

        # test merge_seg
        output = chunk.merge_seg(
            da.zeros(shape, chunks=chunk_size, dtype=np.uint),
            input[bbox[1] : bbox[2] + 1, bbox[3] : bbox[4] + 1, bbox[5] : bbox[6] + 1],
            bbox,
            lambda a, b: a + b,
        ).compute()

        self.assertTrue(np.array_equal(input, output))

    def test_chunk_seed(self):
        dim = 100
        num_points = 100
        chunk_size = (9, 8, 7)
        dtype = float

        points = np.unique(np.random.randint(0, dim, (num_points, 3)), axis=0)
        num_points = points.shape[0]
        pred = np.random.rand(num_points) > 0.5

        gt = np.zeros((dim, dim, dim))
        gt[points[:, 0], points[:, 1], points[:, 2]] = pred

        seeded = chunk.chunk_seed(
            [dim, dim, dim], points, pred, chunk_size, dtype
        ).compute()

        self.assertTrue(np.array_equal(gt, seeded))

    def test_chunk_max_pool(self):
        shape = (100, 100, 100)
        input = generate_simplex_noise(shape, 0.1)
        chunk_size = (9, 8, 7)

        gt = F.max_pool3d(
            torch.from_numpy(input).view(1, 1, *shape),
            chunk_size,
            stride=chunk_size,
            ceil_mode=True,
        )[0, 0]

        downsampled = chunk.chunk(
            chunk._chunk_max_pool,
            [da.from_array(input, chunks=chunk_size)],
            [object],
        ).compute()

        self.assertTrue(np.array_equal(gt.numpy(), downsampled))

    # def test_evaluation(self):
    #     chunk_size = (9, 8, 7)
    #
    #     _, y_true = test_image_nuclei_3d(return_mask=True)
    #     y_pred = rotate(y_true, 2, order=0, reshape=False)
    #
    #     gt_metrics = stardist_matching(y_true, y_pred)
    #     temp = evaluation.get_scores(
    #         da.from_array(y_true, chunks=chunk_size),
    #         da.from_array(y_pred, chunks=chunk_size),
    #     )
    #     pred_metrics = evaluation.matching(*temp).compute()
    #
    #     self.assertTrue(gt_metrics, pred_metrics)
    #
    def test_chunk_unique(self):
        # test both argwhere_seg and simple_chunk's bbox
        shape = (100, 100, 100)
        chunk_size = (9, 8, 7)

        input = np.zeros(shape, dtype=int)
        input[40:60, 40:60, 40:60] = np.random.randint(0, 100, (20, 20, 20))

        gt_unique, gt_inverse = np.unique(input, return_inverse=True)
        unique, inverse = dask.compute(
            *chunk.chunk_unique(
                da.from_array(input, chunks=chunk_size), return_inverse=True
            )
        )
        self.assertTrue(np.array_equal(gt_unique, unique))
        self.assertTrue(np.array_equal(gt_inverse.reshape(shape), inverse[:]))

        unique = chunk.chunk_unique(
            da.from_array(input, chunks=chunk_size), return_inverse=False
        ).compute()
        self.assertTrue(np.array_equal(gt_unique, unique))


if __name__ == "__main__":
    unittest.main()
