# adapted from https://github.com/stardist/stardist/blob/f73cdc44f718d36844b38c1f1662dbb66d157182/stardist/matching.py
import numpy as np
import os
import sys
import h5py

from numba import jit
from scipy.optimize import linear_sum_assignment
from collections import namedtuple

import dask
import dask.array as da
from dask.diagnostics import ProgressBar

import chunk
from utils import dask_read_array, dask_write_array

matching_criteria = dict()


@dask.delayed
def _chunk_relabel(unique):
    remapping = np.arange(len(unique))
    inverse_map = unique

    if len(unique) and unique[0] != 0:
        remapping += 1
        inverse_map = np.concatenate([[0], inverse_map])

    return remapping, inverse_map


def chunk_relabel_sequential(vol):
    # remap integers to lowest possible integers while preserving original order
    # whilst treating 0 as background
    unique, inverse = chunk.chunk_unique(vol, return_inverse=True)
    temp = _chunk_relabel(unique)
    remapping, inverse_map = [
        da.from_delayed(temp[i], shape=[np.nan], dtype=vol.dtype) for i in range(2)
    ]

    remapped = chunk.chunk_remap(inverse, remapping)
    return remapped, inverse_map


@jit(nopython=True)
def _label_overlap(x, y, x_max, y_max):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x_max, 1 + y_max), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return [overlap]


def label_overlap(*args, **kwargs):
    # NOTE: refactor chunk.chunk so that passing block_info into func is optional
    # remove block_info from input
    kwargs.pop("block_info")
    return _label_overlap(*args, **kwargs)


def _safe_divide(x, y, eps=1e-10):
    """computes a safe divide which returns 0 if y is zero"""
    if np.isscalar(x) and np.isscalar(y):
        return x / y if np.abs(y) > eps else 0.0
    else:
        out = np.zeros(np.broadcast(x, y).shape, np.float32)
        np.divide(x, y, out=out, where=np.abs(y) > eps)
        return out


def intersection_over_union(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))


matching_criteria["iou"] = intersection_over_union


def intersection_over_true(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pixels_true)


matching_criteria["iot"] = intersection_over_true


def intersection_over_pred(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return _safe_divide(overlap, n_pixels_pred)


matching_criteria["iop"] = intersection_over_pred


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    # also known as "average precision" (?)
    # -> https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    # also known as "dice coefficient"
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


@dask.delayed
def _aggregate_overlap(overlap):
    stacked = np.stack(overlap.flatten().tolist(), axis=0)
    return np.sum(stacked, axis=0)


def get_scores(y_true, y_pred, criterion="iou"):
    assert y_true.shape == y_pred.shape
    assert criterion in matching_criteria

    y_true, map_rev_true = chunk_relabel_sequential(y_true)
    y_pred, map_rev_pred = chunk_relabel_sequential(y_pred)

    overlap = chunk.chunk(
        label_overlap,
        [y_true, y_pred],
        output_dataset_dtypes=[object],
        x_max=da.max(y_true),
        y_max=da.max(y_pred),
    )

    overlap = da.from_delayed(
        _aggregate_overlap(overlap), shape=[np.nan, np.nan], dtype=float
    )
    scores = dask.delayed(matching_criteria[criterion])(overlap)
    scores = da.from_delayed(scores, shape=overlap.shape, dtype=overlap.dtype)

    return scores, map_rev_true, map_rev_pred, criterion


def matching(
    scores, map_rev_true, map_rev_pred, criterion, thresh=0.5, report_matches=False
):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false negatives (fn)
    whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default IoU)
    report_matches: bool
        if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')

    Returns
    -------
    Matching object with different metrics as attributes

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true,5,axis = 0)

    >>> stats = matching(y_true, y_pred)
    >>> print(stats)
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0, recall=0, accuracy=0, f1=0, n_true=1, n_pred=1, mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

    """
    if thresh is None:
        thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else list(map(float, thresh))

    @dask.delayed
    def _single(scores, map_rev_true, map_rev_pred, thr):
        assert 0 <= np.min(scores) <= np.max(scores) <= 1

        # ignoring background
        scores = scores[1:, 1:]
        n_true, n_pred = scores.shape
        n_matched = min(n_true, n_pred)

        # not_trivial = n_matched > 0 and np.any(scores >= thr)
        not_trivial = n_matched > 0
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2 * n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind, pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        # assert tp+fp == n_pred
        # assert tp+fn == n_true

        # the score sum over all matched objects (tp)
        sum_matched_score = (
            np.sum(scores[true_ind, pred_ind][match_ok]) if not_trivial else 0.0
        )

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score = _safe_divide(sum_matched_score, n_true)
        panoptic_quality = _safe_divide(sum_matched_score, tp + fp / 2 + fn / 2)

        stats_dict = dict(
            criterion=criterion,
            thresh=thr,
            fp=fp,
            tp=tp,
            fn=fn,
            precision=precision(tp, fp, fn),
            recall=recall(tp, fp, fn),
            accuracy=accuracy(tp, fp, fn),
            f1=f1(tp, fp, fn),
            n_true=n_true,
            n_pred=n_pred,
            mean_true_score=mean_true_score,
            mean_matched_score=mean_matched_score,
            panoptic_quality=panoptic_quality,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update(
                    # int() to be json serializable
                    matched_pairs=tuple(
                        (int(map_rev_true[i]), int(map_rev_pred[j]))
                        for i, j in zip(1 + true_ind, 1 + pred_ind)
                    ),
                    matched_scores=tuple(scores[true_ind, pred_ind]),
                    matched_tps=tuple(map(int, np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update(
                    matched_pairs=(),
                    matched_scores=(),
                    matched_tps=(),
                )
        return namedtuple("Matching", stats_dict.keys())(*stats_dict.values())

    if np.isscalar(thresh):
        thresh = [thresh]

    result = [_single(scores, map_rev_true, map_rev_pred, x) for x in thresh]

    return result if len(result) > 1 else result[0]


def main(base_path, id, h5):
    if h5 == "inference":
        h5 = f"inferred_{id}.h5"
    elif h5 == "baseline":
        h5 = f"seg_{id}.h5"
    else:
        raise ValueError("h5 must be either 'inference' or 'baseline'")

    pred = h5py.File(os.path.join(base_path, h5)).get("seg")
    pred = dask_read_array(pred)
    # map trunk to background
    pred = pred * (pred > 1)

    gt = h5py.File(os.path.join(base_path, f"gt_{id}.h5")).get("main")
    gt = dask_read_array(gt)

    scores, map_rev_true, map_rev_pred, criterion = dask.compute(
        *get_scores(gt, pred, criterion="iou"), scheduler="single-threaded"
    )

    file = h5py.File(os.path.join(base_path, "scores_" + h5), "w")
    file.create_dataset("scores", data=scores)
    file.create_dataset("map_rev_true", data=map_rev_true)
    file.create_dataset("map_rev_pred", data=map_rev_pred)
    file.attrs["criterion"] = criterion

    file.close()


if __name__ == "__main__":
    with ProgressBar():
        main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
