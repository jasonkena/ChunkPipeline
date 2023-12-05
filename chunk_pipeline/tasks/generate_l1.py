import subprocess
import tempfile
import os
import logging

import numpy as np
from cloudvolume import Skeleton
import dask

import open3d as o3d
import kimimaro

from chunk_pipeline.tasks.generate_skeleton import _longest_path


def parse_skel(filename):
    result = {}
    lines = open(filename).readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]

    assert lines[0].startswith("ON")
    num_original = int(lines[0].split()[1])
    lines = lines[1:]
    result["original"] = np.stack(
        [np.array([float(x) for x in line.split()]) for line in lines[:num_original]],
        axis=0,
    )
    lines = lines[num_original:]

    # NOTE: samples are potentially inf
    assert lines[0].startswith("SN")
    num_sampled = int(lines[0].split()[1])
    lines = lines[1:]
    result["sample"] = np.stack(
        [np.array([float(x) for x in line.split()]) for line in lines[:num_sampled]],
        axis=0,
    )
    lines = lines[num_sampled:]

    assert lines[0].startswith("CN")
    num_branches = int(lines[0].split()[1])
    lines = lines[1:]

    branches = []
    for _ in range(num_branches):
        assert lines[0].startswith("CNN")
        num_nodes = int(lines[0].split()[1])
        lines = lines[1:]
        branches.append(
            np.stack(
                [
                    np.array([float(x) for x in line.split()])
                    for line in lines[:num_nodes]
                ],
                axis=0,
            )
        )
        lines = lines[num_nodes:]
    result["branches"] = branches
    len_branches = [x.shape[0] for x in branches]

    assert lines[0] == "EN 0"
    lines = lines[1:]
    assert lines[0] == "BN 0"
    lines = lines[1:]

    assert lines[0].startswith("S_onedge")
    lines = lines[1:]
    result["sample_onedge"] = np.array(list(map(int, lines[0].split()))) > 0
    lines = lines[1:]

    assert lines[0].startswith("GroupID")
    lines = lines[1:]
    result["sample_groupid"] = np.array(list(map(int, lines[0].split())))
    lines = lines[1:]

    # flattened branches
    assert lines[0].startswith("SkelRadius")
    lines = lines[1:]
    result["branches_skelradius"] = np.split(
        np.array(list(map(float, lines[0].split()))), np.cumsum(len_branches)
    )[:-1]
    lines = lines[1:]

    assert lines[0].startswith("Confidence_Sigma")
    lines = lines[1:]
    result["sample_confidence_sigma"] = np.array(list(map(float, lines[0].split())))
    lines = lines[1:]

    assert lines[0] == "SkelRadius2 0"
    lines = lines[1:]
    assert lines[0] == "Alpha 0"
    lines = lines[1:]

    assert lines[0].startswith("Sample_isVirtual")
    lines = lines[1:]
    result["sample_isvirtual"] = np.array(list(map(int, lines[0].split()))) > 0
    lines = lines[1:]

    assert lines[0].startswith("Sample_isBranch")
    lines = lines[1:]
    result["sample_isbranch"] = np.array(list(map(int, lines[0].split()))) > 0
    lines = lines[1:]

    assert lines[0].startswith("Sample_radius")
    lines = lines[2:]

    assert lines[0].startswith("Skel_isVirtual")
    lines = lines[1:]
    result["skel_isvirtual"] = np.split(
        np.array(list(map(int, lines[0].split()))) > 0, np.cumsum(len_branches)
    )[:-1]
    lines = lines[1:]

    # NOTE: this does not generate anything useful, as samples are potentially inf
    assert lines[0].startswith("Corresponding_sample_index")
    lines = lines[1:]
    result["corresponding_sample_index"] = np.split(
        np.array(list(map(int, lines[0].split()))), np.cumsum(len_branches)
    )[:-1]
    lines = lines[1:]

    assert len(lines) == 0

    return result


def to_cloud_volume_skeleton(parsed):
    branch_length = np.array([len(x) for x in parsed["branches"]])
    flattened_vertices = np.concatenate(parsed["branches"])
    flattened_radii = np.concatenate(parsed["branches_skelradius"])

    # NOTE: may need to replace this with np.isclose to allow within epsilon dists
    unique, index, inverse = np.unique(
        flattened_vertices, axis=0, return_index=True, return_inverse=True
    )
    edges = []

    branch_inverse = np.split(inverse, np.cumsum(branch_length))[:-1]
    for branch in branch_inverse:
        edges.append(np.stack([branch[:-1], branch[1:]], axis=1))
    edges = np.concatenate(edges, axis=0)

    flattened_radii = flattened_radii[np.argsort(inverse)]
    radii = np.split(flattened_radii, np.cumsum(branch_length))[:-1]
    assert max([len(np.unique(x)) for x in radii])
    radii = flattened_radii[index]

    skel = Skeleton(vertices=unique, edges=edges, radii=radii)

    return skel


def point_cloud_to_ply(pc, out_filename):
    # NOTE: assumes isotropic data, properly scaled data
    # pc: [N, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    o3d.io.write_point_cloud(out_filename, pcd)

    return out_filename


def calculate_downscale_factor(anisotropy, num_orig_points, num_downsampled_points):
    # inspired by getInitRadiuse in DataMgr.cpp
    # assuming isotropic data
    cgrid = 0.2
    unit_size = np.sqrt(np.sum(anisotropy) ** 2)
    new_unit_size = unit_size * (num_orig_points / num_downsampled_points) ** (
        1.0 / 3.0
    )
    downscale_factor = cgrid / new_unit_size

    # print("new_unit_size", new_unit_size)
    # print("downscale_factor", downscale_factor)

    return downscale_factor


@dask.delayed
def generate_l1_from_vol(vol, idx, *args, **kwargs):
    # assumes isotropic volume if aniostropy is None

    kwargs = kwargs.copy()

    pc = np.argwhere(vol == idx)
    skel = generate_l1(pc, *args, **kwargs)

    return skel


@dask.delayed
def generate_l1_from_npz(filename, idx, input_key, seg_key, *args, **kwargs):
    data = np.load(filename)
    pc = data[input_key][data[seg_key] == idx].astype(np.float64)

    skel = generate_l1(pc, *args, **kwargs)

    return skel


@dask.delayed
def generate_l1_from_pc(pc, *args, **kwargs):
    # assumes pc is still dense anisotropic, returns anisotropic skeleton

    # cgrid radius is used as initial radius
    pc = pc.astype(np.float64)

    skel = generate_l1(pc, *args, **kwargs)

    return skel


def skel_path_length(skel):
    # assuming isotropic skeleton
    # return l2 distance between all edges
    v = skel.vertices
    e = skel.edges
    return np.sum(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1))


def generate_l1(
    pc,
    bin_path,
    json_path,
    tmp_dir,
    store_tmp,
    # downscale_factor,
    noise_std,
    num_sample,
    percent_sample=1,
    max_errors=5,
    error_upsample=1.5,
    anisotropy=(1, 1, 1),
):
    # NOTE: assumes anisotropic input, will multiply by anisotropy to get anisotropic data
    # on parse errors (skeleton not fully formed), undo some of the downscaling
    if num_sample > 0:
        num_sample = int(min(pc.shape[0], num_sample, pc.shape[0] * percent_sample))

    if len(pc) == 0:
        return Skeleton()

    num_orig_points = pc.shape[0]
    np.random.seed(0)
    if (num_sample > 0) and (pc.shape[0] > num_sample):
        choice = np.random.choice(pc.shape[0], num_sample, replace=False)
        pc = pc[choice]
    num_downsampled_points = pc.shape[0]

    anisotropy = np.array(anisotropy)
    pc *= anisotropy
    # apply noise to isotropic PC
    pc = pc + np.random.normal(0, noise_std, pc.shape)

    downscale_factor = calculate_downscale_factor(
        anisotropy, num_orig_points, num_downsampled_points
    )
    pc *= downscale_factor

    error_count = 0
    best_skel = None
    best_skel_length = -1
    # best_skel_n_components = float("inf")

    # NOTE: TODO: delegate each try to dask
    while error_count <= max_errors:
        with tempfile.NamedTemporaryFile(
            suffix=".ply", dir=tmp_dir, delete=(not store_tmp)
        ) as tmp_ply, tempfile.NamedTemporaryFile(
            suffix=".txt", dir=tmp_dir, delete=(not store_tmp)
        ) as tmp_log, tempfile.NamedTemporaryFile(
            suffix=".skel", dir=tmp_dir, delete=(not store_tmp)
        ) as tmp_skel:
            ply_path = point_cloud_to_ply(pc, tmp_ply.name)
            skel_path = tmp_skel.name
            cmd = f"{bin_path} {ply_path} {skel_path} {json_path}"

            print(f"Running command: {cmd}")
            print(f"Logging to: {tmp_log.name}")

            # NOTE: this is a blocking call, can use subprocess.Popen to run in background
            call = subprocess.run(cmd.split(), stdout=tmp_log, stderr=tmp_log)
            if call.returncode != 0:
                logging.warning(
                    f"L1 skeletonization failed {ply_path} {skel_path} {json_path}"
                )

            skel = None
            try:
                skel = parse_skel(skel_path)
            except Exception:
                logging.warning(
                    f"L1 parsing failed {ply_path} {skel_path} {json_path} attempt {error_count}, retrying"
                )
            if skel is not None:
                skel = to_cloud_volume_skeleton(skel)
                # now in isotropic real (nm) units
                skel.vertices /= downscale_factor
                # calculate path length (with potentially multiple connected components)
                path_length = skel_path_length(skel)
                # merge connected components while still in isotropic coordinates
                skel = kimimaro.join_close_components(skel, radius=None)
                # now in idx units
                skel.vertices /= anisotropy

                assert path_length > 0
                if path_length > best_skel_length:
                    best_skel = skel
                    best_skel_length = path_length
                # n_components = len(skel.components())
                # assert n_components > 0
                #
                # if n_components < best_skel_n_components:
                #     best_skel = skel
                #     best_skel_n_components = n_components
                # if n_components == 1:
                #     break

            error_count += 1
            pc = pc * error_upsample
            downscale_factor *= error_upsample

    if best_skel is None:
        logging.warning(
            f"L1 parsing failed {ply_path} {skel_path} {json_path}, saving blank"
        )
        return Skeleton()

    return best_skel


def task_generate_l1_from_vol(cfg, vols):
    # NOTE: this assumes isotropic volume
    general = cfg["GENERAL"]
    l1 = cfg["L1"]

    results = {}
    for vol_idx in vols:
        results[vol_idx] = {}
        for rib_idx in l1["IDS"]:
            results[vol_idx][rib_idx] = generate_l1_from_vol(
                vols[vol_idx],
                rib_idx,
                l1["BIN_PATH"],
                l1["JSON_PATH"],
                l1["TMP_DIR"],
                l1["STORE_TMP"],
                # l1["DOWNSCALE_FACTOR"],
                l1["NOISE_STD"],
                l1["NUM_SAMPLE"],
                max_errors=l1["MAX_ERRORS"],
                error_upsample=l1["ERROR_UPSAMPLE"],
            )
    return results


def task_generate_snemi_l1_from_vol(cfg, vols):
    general = cfg["GENERAL"]
    anisotropy = general["ANISOTROPY"]
    l1 = cfg["L1"]

    results = {}
    assert len(vols) == 1
    vol = list(vols.values())[0]
    for idx in l1["IDS"]:
        results[idx] = generate_l1_from_vol(
            vol,
            idx,
            l1["BIN_PATH"],
            l1["JSON_PATH"],
            l1["TMP_DIR"],
            l1["STORE_TMP"],
            # l1["DOWNSCALE_FACTOR"],
            l1["NOISE_STD"],
            l1["NUM_SAMPLE"],
            max_errors=l1["MAX_ERRORS"],
            error_upsample=l1["ERROR_UPSAMPLE"],
            anisotropy=anisotropy,
        )
    return results


def task_generate_l1_from_pc(cfg, pc):
    # identical signature with task_skeletonize from generate_skeleton.py
    general = cfg["GENERAL"]
    l1 = cfg["L1"]

    anisotropy = general["ANISOTROPY"]
    idx = pc["idx"]

    # l1["STORE_TMP"] = True
    skel = generate_l1_from_pc(
        idx,
        l1["BIN_PATH"],
        l1["JSON_PATH"],
        l1["TMP_DIR"],
        l1["STORE_TMP"],
        # l1["DOWNSCALE_FACTOR"],
        l1["NOISE_STD"],
        l1["NUM_SAMPLE"],
        max_errors=l1["MAX_ERRORS"],
        error_upsample=l1["ERROR_UPSAMPLE"],
        anisotropy=anisotropy,
    )
    longest_path = _longest_path(skel)
    result = {"skeleton": skel, "longest_path": longest_path}

    # __import__('pdb').set_trace()
    # result = dask.compute(result, scheduler="single-threaded")
    # __import__('pdb').set_trace()
    # np.save("/mmfs1/data/adhinart/dumb/welp.npy", result)
    # print("saved")

    return result


def task_generate_l1_from_npz(cfg):
    # identical signature with task_skeletonize from generate_skeleton.py
    general = cfg["GENERAL"]
    l1 = cfg["L1"]
    npz = cfg["NPZ"]

    # l1["STORE_TMP"] = True
    results = {}
    for key in npz:
        filename, id, input_key, seg_key = npz[key]
        skel = generate_l1_from_npz(
            filename,
            id,
            input_key,
            seg_key,
            l1["BIN_PATH"],
            l1["JSON_PATH"],
            l1["TMP_DIR"],
            l1["STORE_TMP"],
            # l1["DOWNSCALE_FACTOR"],
            l1["NOISE_STD"],
            l1["NUM_SAMPLE"],
            max_errors=l1["MAX_ERRORS"],
            error_upsample=l1["ERROR_UPSAMPLE"],
        )
        longest_path = _longest_path(skel)
        results[key] = {"skeleton": skel, "longest_path": longest_path}

    # __import__('pdb').set_trace()
    # result = dask.compute(result, scheduler="single-threaded")
    # __import__('pdb').set_trace()
    # np.save("/mmfs1/data/adhinart/dumb/welp.npy", result)
    # print("saved")

    return results
