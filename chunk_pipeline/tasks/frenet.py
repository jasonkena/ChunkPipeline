import numpy as np
from math import gcd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.interpolate import UnivariateSpline

import dask
import dask.array as da
import networkx as nx

from chunk_pipeline.utils import object_array


def nx_from_skel(skel):
    edges = []
    for a, b in skel.edges:
        l2 = np.linalg.norm(skel.vertices[a] - skel.vertices[b])
        edges.append((a, b, l2))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


def get_trunk_path(skel):
    G = nx_from_skel(skel)
    assert nx.is_connected(G)

    unique, counts = np.unique(skel.edges.reshape(-1), return_counts=True)
    n_edges = np.ones(len(skel.vertices))
    n_edges[unique] = counts

    end_points = np.where(n_edges == 1)[0]
    is_intersection = n_edges > 2

    paths = []

    assert len(end_points) >= 2
    for i, a in enumerate(end_points[:-1]):
        for j, b in enumerate(end_points[i + 1 :]):
            try:  # since unique != np.arange(skel.vertices.shape[0]) for some reason
                # can probably avoid calling this twice
                path = nx.shortest_path(G, a, b, weight="weight")
                length = nx.shortest_path_length(G, a, b, weight="weight")
                # number of intersections as primary, length as tie breaker
                weight = (np.sum(is_intersection[path]), length)

                paths.append((path, weight))
            except:
                pass
    paths = sorted(paths, key=lambda x: x[1])
    path = paths[-1][0]
    assert max(path) < len(skel.vertices)

    return path


def closest_trunk_idx(skel, trunk_path):
    # for each point in skeleton, get idx to closest point in trunk (by path length)
    # returns order in trunk_path, not the idx of the point in the skeleton
    G = nx_from_skel(skel)
    # get all paths, using trunk_path ids as sources
    paths = nx.multi_source_dijkstra_path(G, trunk_path)
    assert len(paths) == len(skel.vertices)
    closest_idx = []
    for i in range(len(skel.vertices)):
        # start of path is source
        closest_idx.append(paths[i][0])
    # closest_idx in idx of trunk point wrt skel
    closest_idx = np.array(closest_idx)
    remap = -np.ones(len(skel.vertices), dtype=int)
    remap[trunk_path] = np.arange(len(trunk_path))

    # closest_idx in idx of trunk point wrt trunk_path
    closest_idx = remap[closest_idx]
    assert np.all(closest_idx != -1)

    return closest_idx


def closest_centerline(skel, centerline, trunk_path):
    # NOTE: assumes isotropic skel
    # for each point in reference: [centerline + (skel - trunk_path)], get idx to closest point in centerline (by both l2 distance and graph distance)
    is_trunk = np.zeros(len(skel.vertices), dtype=bool)
    is_trunk[trunk_path] = True
    reference = np.concatenate([centerline, skel.vertices[~is_trunk]], axis=0)

    assert len(trunk_path) <= skel.vertices.shape[0]
    closest_idx_centerline_to_centerline = np.arange(centerline.shape[0])
    # has length rest
    closest_idx_rest_to_trunk = closest_trunk_idx(skel, trunk_path)[~is_trunk]
    # has length trunk
    closest_idx_trunk_to_centerline = get_closest(
        skel.vertices[trunk_path], centerline
    )[1]
    closest_idx_rest_to_centerline = closest_idx_trunk_to_centerline[
        closest_idx_rest_to_trunk
    ]
    closest_idx_reference_to_centerline = np.concatenate(
        [closest_idx_centerline_to_centerline, closest_idx_rest_to_centerline], axis=0
    )

    return reference, closest_idx_reference_to_centerline


@dask.delayed
def get_centerline(skel, path_length):
    # NOTE: assumes that skeleton is already "beautified" and isotropic
    # returns interpolated centerline, with trunk_path with idx of non-interpolated skel.vertices
    trunk_path = get_trunk_path(skel)

    centerline = spline_interpolate_centerline(skel.vertices[trunk_path], path_length)
    assert max(trunk_path) < len(skel.vertices)

    return centerline, trunk_path


@dask.delayed
def segment_pc(idx, spine, seg, centerline, path_length, trunk_path, anisotropy, skel):
    # centerline is interpolated
    dtype = np.float64
    anisotropy = np.array(anisotropy).astype(dtype)
    # idx: [n, 3]
    # spine: [n]

    # get isotropic pc
    # [n, 5] (idx, spine [bool], seg)
    pc = np.concatenate(
        [idx.astype(dtype) * anisotropy, spine[:, None], seg[:, None]], axis=1
    )

    # NOTE: this assumes isotropic skel and centerline
    skel.vertices = skel.vertices * anisotropy
    reference, closest_idx_reference_to_centerline = closest_centerline(
        skel, centerline, trunk_path
    )

    dist, closest_idx = get_closest(pc[:, :3], reference)
    # maps to closest centerline
    closest_idx = closest_idx_reference_to_centerline[closest_idx]

    assert centerline.shape[0] == path_length
    assert np.max(closest_idx) < path_length

    unique, inverse, counts = np.unique(
        closest_idx, return_inverse=True, return_counts=True
    )
    idx = np.argsort(inverse)

    closest_idx = closest_idx[idx]
    dist = dist[idx]
    pc = pc[idx]

    cumsum = np.zeros(path_length, dtype=int)
    cumsum[unique] = counts
    cumsum = np.cumsum(cumsum)

    # split at every idx along longest_path
    split_idx = np.arange(0, path_length, 1)
    split_idx = cumsum[split_idx]

    pc_segments = np.split(pc, split_idx[1:])
    closest_idx_segments = np.split(closest_idx, split_idx[1:])
    dist_segments = np.split(dist, split_idx[1:])

    # assert min([len(x) for x in pc_segments]) > 0

    # returns isotropic reference
    result = {
        "reference": reference,
        "split": [],
    }

    for pc, closest_idx, dist in zip(pc_segments, closest_idx_segments, dist_segments):
        # potentially empty
        result["split"].append({"pc": pc, "dist": dist, "closest_idx": closest_idx})

    result["split"] = object_array(result["split"])

    return result


def cylindrical_segment_pc(split, centerline, T, N, B, dis_geo_skel):
    split = split.item()
    pc = split["pc"]
    dist = split["dist"]
    closest_idx = split["closest_idx"]

    if pc.shape[0] == 0:
        # return empty array
        return object_array([pc])

    cyd_pc = cylindrical_transformation(
        pc, centerline, dist, closest_idx, T, N, B, dis_geo_skel
    )
    assert cyd_pc.shape == pc.shape

    return object_array([cyd_pc])


def get_closest(pc_a, pc_b):
    # pc_a: (m,3)
    # pc_b: (n,3)
    # return: (m) - the index of the closest point in pc_b for each point in pc_a

    # pc_a = pc_a.astype(np.float64)
    # pc_b = pc_b.astype(np.float64)

    tree = KDTree(pc_b)
    dist, idx = tree.query(pc_a, workers=-1)

    if np.max(idx) >= pc_b.shape[0]:
        np.save("/mmfs1/data/adhinart/dendrite/logs/pc_a.npy", pc_a)
        np.save("/mmfs1/data/adhinart/dendrite/logs/pc_b.npy", pc_b)
        raise ValueError("idx is out of range")

    return dist, idx


def spline_interpolate_centerline(path, path_length, kwargs={"k": 2, "s": 15}):
    # skel_upsampled = (path,k=2,s=15)
    # path: (n,3)
    # path_length: int
    # kwargs: dict, for UnivariateSpline
    l2 = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cumsum = np.cumsum(l2)
    cumsum = np.insert(cumsum, 0, 0)

    total_length = cumsum[-1]
    x, y, z = path[:, 0], path[:, 1], path[:, 2]
    new_path = np.zeros((path_length, 3))

    w = cumsum
    try:
        sx = UnivariateSpline(w, x, **kwargs)
        sy = UnivariateSpline(w, y, **kwargs)
        sz = UnivariateSpline(w, z, **kwargs)
    except Exception as e:
        raise ValueError(f"Error in spline interpolation: {e} {cumsum}")

    wnew = np.linspace(0, total_length, num=path_length)
    new_path[:, 0] = sx(wnew)
    new_path[:, 1] = sy(wnew)
    new_path[:, 2] = sz(wnew)

    return new_path


def interp_centerline(path, path_length):
    # path: (n,3)
    # path_length: int

    l2 = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cumsum = np.cumsum(l2)
    cumsum = np.insert(cumsum, 0, 0)
    total_length = cumsum[-1]

    sample_length = np.linspace(0, total_length, path_length)

    points = [np.interp(sample_length, cumsum, path[:, i]) for i in range(3)]
    points = np.stack(points, axis=1)

    return points


def get_cord_skel(skel):
    T, N, B = frenet_frame(skel)
    dis_geo_skel_tmp = np.insert((((T**2).sum(1)) ** 0.5)[:-1], 0, 0)
    dis_geo_skel = np.zeros_like(dis_geo_skel_tmp)
    for i in range(dis_geo_skel.shape[0]):
        dis_geo_skel[i] = dis_geo_skel_tmp[: i + 1].sum()
    cord_skel = np.concatenate(
        (dis_geo_skel[:, None], np.zeros(dis_geo_skel.shape)[:, None]), axis=1
    )
    cord_skel = np.concatenate(
        (cord_skel, np.zeros(dis_geo_skel.shape)[:, None]), axis=1
    )

    return cord_skel, T, N, B, dis_geo_skel


def cylindrical_transformation(pc, skel, dist, closest_idx, T, N, B, dis_geo_skel):
    # input pc [N, 3+1], smoothed_skel [S,3]

    pc_skel = skel[closest_idx]
    vec_tan = T[closest_idx]
    vec_norm = N[closest_idx]
    vec_binorm = B[closest_idx]

    # non-skeleton
    dis_geo_pc = dis_geo_skel[closest_idx]

    t = ((pc_skel * vec_tan).sum(1) - (pc[:, :3] * vec_tan).sum(1)) / (
        (vec_tan**2).sum(1)
    )
    pc_proj = pc[:, :3] + t[:, None] * vec_tan
    vec_proj = pc_proj - pc_skel
    cos_norm = (vec_norm * vec_proj).sum(1) / (
        (vec_norm**2).sum(1) + (vec_proj**2).sum(1)
    )
    cos_binorm = (vec_binorm * vec_proj).sum(1) / (
        (vec_binorm**2).sum(1) + (vec_proj**2).sum(1)
    )
    cos_binorm[cos_binorm >= 0] = 1
    cos_binorm[cos_binorm < 0] = -1
    dis_theta_pc = np.arccos(cos_norm) * cos_binorm

    cyd_pc = np.concatenate((dis_geo_pc[:, None], dist[:, None]), axis=1)
    cyd_pc = np.concatenate((cyd_pc, dis_theta_pc[:, None]), axis=1)
    cyd_pc = np.concatenate((cyd_pc, pc[:, 3:]), axis=1)

    return cyd_pc


def frenet_frame(skeleton):
    # tangent
    skel_tan = np.zeros(skeleton.shape)
    skel_tan[:-1] = skeleton[1:] - skeleton[:-1]
    skel_tan[-1] = skel_tan[-2]
    # normal
    v_1 = skeleton[:-2]
    v_2 = skeleton[1:-1]
    v_3 = skeleton[2:]
    assert np.any(((v_2 - v_1) ** 2).sum(1) != 0)
    t = ((v_2 - v_1) * (v_3 - v_2)).sum(1) / ((v_2 - v_1) ** 2).sum(1)
    assert np.any(t >= 0)
    normal = np.zeros(skeleton.shape)
    normal[:-2] = (v_3 - v_2) - t[:, None] * (v_2 - v_1)

    # find the last non-zero vertex
    idx_last_non_zero_vert = np.argwhere(
        (normal[:, 0] != 0) + (normal[:, 1] != 0) + (normal[:, 2] != 0)
    )
    if idx_last_non_zero_vert.shape[0] != 0:
        idx_last_non_zero_vert = idx_last_non_zero_vert[-1, 0]
    else:
        print("all vertices are colinear, return [0]")
        return skel_tan, np.zeros([0]), None
    normal[idx_last_non_zero_vert + 1] = normal_backwards(
        skeleton, idx_last_non_zero_vert
    )
    normal[idx_last_non_zero_vert + 2 :] = normal[idx_last_non_zero_vert + 1]
    # intermediate Zero values
    # backward calculate normal
    idx_v_line = np.argwhere(
        (normal[:, 0] == 0) * (normal[:, 1] == 0) * (normal[:, 2] == 0)
    )[:, 0]

    if idx_v_line.shape[0] != 0:
        for idx in idx_v_line[::-1]:
            if idx - 1 >= 0:
                normal[idx] = normal_backwards(skeleton, idx - 1)

    # backward assign normal - last vertice
    idx_v_line = np.argwhere(
        (normal[:, 0] == 0) * (normal[:, 1] == 0) * (normal[:, 2] == 0)
    )[:, 0]

    if idx_v_line.shape[0] != 0:
        if idx_v_line[-1] + 1 == normal.shape[0]:
            idx = idx_v_line[-1] - 1
            while idx in idx_v_line[::-1]:
                idx -= 1
            normal[idx:,] = normal[idx,]  # optimize
    # backward assign normal - intermediate vertice
    idx_v_line = np.argwhere(
        (normal[:, 0] == 0) * (normal[:, 1] == 0) * (normal[:, 2] == 0)
    )[:, 0]
    if idx_v_line.shape[0] != 0:
        for idx in idx_v_line[::-1]:
            normal[idx] = normal[idx + 1]
    # binormal
    skel_binorm = np.cross(skel_tan, normal)
    assert (
        np.argwhere(
            (skel_binorm[:, 0] == 0)
            * (skel_binorm[:, 1] == 0)
            * (skel_binorm[:, 2] == 0)
        ).shape[0]
        == 0
    )
    return skel_tan, normal, skel_binorm


def normal_backwards(skeleton, idx_1):
    # normal_backward
    assert np.any(((skeleton[idx_1 + 2] - skeleton[idx_1 + 1]) ** 2).sum() != 0)
    lda = (
        (skeleton[idx_1 + 1] - skeleton[idx_1])
        * (skeleton[idx_1 + 2] - skeleton[idx_1 + 1])
    ).sum() / ((skeleton[idx_1 + 2] - skeleton[idx_1 + 1]) ** 2).sum()
    return (
        skeleton[idx_1]
        - skeleton[idx_1 + 1]
        - lda * (skeleton[idx_1 + 1] - skeleton[idx_1 + 2])
    )


@dask.delayed
def merge_combined(array):
    pc = np.concatenate([x["pc"] for x in array], axis=0)
    dist = np.concatenate([x["dist"] for x in array], axis=0)
    closest_idx = np.concatenate([x["closest_idx"] for x in array], axis=0)

    if len(pc) == 0:
        # return negative one if segment is empty
        pc = -np.ones((1, 5))
        dist = -np.ones((1,))
        closest_idx = -np.ones((1,))
        return {"pc": pc, "dist": dist, "closest_idx": closest_idx}

    return {"pc": pc, "dist": dist, "closest_idx": closest_idx}


def stride_segments(combined, centerline, window_length, stride_length):
    # centerline should already be computed (not delayed)
    l2 = np.linalg.norm(centerline[1:] - centerline[:-1], axis=1)
    cumsum = np.cumsum(l2)
    cumsum = np.insert(cumsum, 0, 0)
    total_length = cumsum[-1]

    segments = []
    idx = []

    break_now = False
    for start in np.arange(0, total_length, stride_length):
        end = start + window_length
        if end < total_length:
            left = np.searchsorted(cumsum, start, side="left")
            right = np.searchsorted(cumsum, end, side="right")
        else:
            start = total_length - window_length
            left = np.searchsorted(cumsum, start, side="left")
            right = cumsum.shape[0]
            break_now = True

        segments.append(combined[left:right])
        if len(combined[left:right]) == 0:
            print("empty segment")
            __import__("pdb").set_trace()
        idx.append([left, right])
        if break_now:
            break

    return segments, idx


def beautify_skel(skel, anisotropy, sigma_threshold=3):
    # NOTE: assumes anisotropic skeleton, returns isotropic skeleton
    # NOTE: may modify skel in-place
    dtype = np.float64
    anisotropy = np.array(anisotropy).astype(dtype)
    skel.vertices = skel.vertices.astype(dtype) * anisotropy

    # replaces edges larger than sigma_threshold * median edge with a straight line with path_length = median edge (approximately)
    path_lengths = np.linalg.norm(
        skel.vertices[skel.edges[:, 0]] - skel.vertices[skel.edges[:, 1]], axis=1
    )
    median = np.median(path_lengths)
    thresh = median * sigma_threshold
    edge_ids = np.where(path_lengths > thresh)[0]

    if len(edge_ids) == 0:
        return skel

    for edge_id in edge_ids:
        # don't delete edges, modify them in-place and concatenate new edges
        edge = skel.edges[edge_id]
        pt1, pt2 = skel.vertices[edge[0]], skel.vertices[edge[1]]
        # includes endpoints
        num_pts = int(np.ceil(path_lengths[edge_id] / median)) + 1
        assert num_pts >= 3  # probably means sigma_threshold should be larger
        # don't include endpoints
        new_pts = np.linspace(pt1, pt2, num_pts)[1:-1]
        new_radii = np.linspace(skel.radius[edge[0]], skel.radius[edge[1]], num_pts)[
            1:-1
        ]
        N = len(skel.vertices)
        skel.vertices = np.concatenate([skel.vertices, new_pts])
        skel.radius = np.concatenate([skel.radius, new_radii])
        skel.vertex_types = np.concatenate([skel.vertex_types, np.zeros(len(new_pts))])
        new_edges = []
        for i in range(N, N + len(new_pts) - 1):
            new_edges.append([i, i + 1])
        new_edges.append([N + len(new_pts) - 1, edge[1]])
        skel.edges[edge_id][1] = N
        new_edges = np.array(new_edges).astype(skel.edges.dtype)
        skel.edges = np.concatenate([skel.edges, new_edges])

    assert len(skel.components()) == 1

    # NOTE: isotropic skel
    return skel

# NOTE: here here here
def get_skel_is_trunk():
    pass


def task_generate_point_cloud_segments(cfg, pc, _skel):
    _skel = _skel["skeleton"]
    idx, spine, seg = pc["idx"], pc["spine"], pc["seg"]

    general = cfg["GENERAL"]
    # uint_dtype = general["UINT_DTYPE"]
    anisotropy = general["ANISOTROPY"]
    isotropic_skel = beautify_skel(_skel, anisotropy)

    chunk_size = general["CHUNK_SIZE"]
    chunk_size = np.prod(chunk_size)

    frenet = cfg["FRENET"]
    path_length = frenet["PATH_LENGTH"]
    window_length = frenet["WINDOW_LENGTH"]
    stride_length = frenet["STRIDE_LENGTH"]
    # segment_per = frenet["SEGMENT_PER"]

    centerline_results = get_centerline(isotropic_skel, path_length)
    centerline, trunk_path = centerline_results[0], centerline_results[1]

    # ceil
    # num_segments = np.ceil(path_length / segment_per).astype(int)
    output = segment_pc(
        idx, spine, seg, centerline, path_length, trunk_path, anisotropy, isotropic_skel
    )

    # FIX THIS
    result = {
        "reference": output["reference"],
        "centerline": centerline,
        # "skel": centerline,
    }

    splitted = da.from_delayed(
        output["split"], shape=(path_length,), dtype=object
    ).rechunk((1,))

    # need to compute to calculate chunks
    centerline = centerline.compute()

    segments, split_idx = stride_segments(
        splitted, centerline, window_length, stride_length
    )

    result["split_idx"] = split_idx

    for i, segment in enumerate(segments):
        merged = merge_combined(segment)
        result[f"pc_{i}"] = da.from_delayed(
            merged["pc"], shape=(np.nan, 5), dtype=np.float64
        )
        result[f"dist_{i}"] = da.from_delayed(
            merged["dist"], shape=(np.nan,), dtype=np.float64
        )
        result[f"closest_idx_{i}"] = da.from_delayed(
            merged["closest_idx"], shape=(np.nan,), dtype=int
        )

    # result["skel"].compute(scheduler="single-threaded")

    return result
