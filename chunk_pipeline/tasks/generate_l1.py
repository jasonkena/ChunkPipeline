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


@dask.delayed
def generate_l1_from_vol(vol, idx, *args, **kwargs):
    # assumes isotropic volume

    pc = np.argwhere(vol == idx)
    return generate_l1(pc, *args, **kwargs)


@dask.delayed
def generate_l1_from_pc(pc, anisotropy, *args, **kwargs):
    # assumes pc is still anisotropic, returns anisotropic skeleton

    anisotropy = np.array(anisotropy)
    # unit_size = np.sqrt(np.sum(anisotropy ** 2))
    unit_size = 1  # cgrid radius is used as initial radius
    pc = pc.astype(np.float64) * anisotropy / unit_size

    skel = generate_l1(pc, *args, **kwargs)
    skel.vertices = skel.vertices * unit_size / anisotropy

    return skel


def generate_l1(
    pc,
    bin_path,
    json_path,
    tmp_dir,
    store_tmp,
    downscale_factor,
    noise_std,
    num_sample,
    percent_sample=0.01,
    max_errors=5,
    error_upsample=1.5,
):
    # on parse errors (skeleton not fully formed), undo some of the downscaling
    if num_sample > 0:
        num_sample = int(min(pc.shape[0], num_sample, pc.shape[0] * percent_sample))

    if len(pc) == 0:
        return Skeleton()

    np.random.seed(0)
    if (num_sample > 0) and (pc.shape[0] > num_sample):
        choice = np.random.choice(pc.shape[0], num_sample, replace=False)
        pc = pc[choice]

    pc = pc + np.random.normal(0, noise_std, pc.shape)
    pc *= downscale_factor

    error_count = 0
    while True:
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
                raise Exception(
                    f"L1 skeletonization failed {ply_path} {skel_path} {json_path}"
                )

            try:
                skel = parse_skel(skel_path)
                break
            except Exception:
                if error_count > max_errors:
                    raise Exception(
                        f"L1 parsing failed {ply_path} {skel_path} {json_path}, retrying"
                    )
                else:
                    logging.warning(
                        f"L1 parsing failed {ply_path} {skel_path} {json_path} attempt {error_count}, retrying"
                    )

                    error_count += 1
                    pc = pc * error_upsample
                    downscale_factor *= error_upsample

    skeleton = to_cloud_volume_skeleton(skel)
    skeleton.vertices /= downscale_factor

    skeleton = kimimaro.join_close_components(skeleton, radius=None)

    return skeleton


def task_generate_l1_from_vol(cfg, vols):
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
                l1["DOWNSCALE_FACTOR"],
                l1["NOISE_STD"],
                l1["NUM_SAMPLE"],
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
        anisotropy,
        l1["BIN_PATH"],
        l1["JSON_PATH"],
        l1["TMP_DIR"],
        l1["STORE_TMP"],
        l1["DOWNSCALE_FACTOR"],
        l1["NOISE_STD"],
        l1["NUM_SAMPLE"],
    )
    longest_path = _longest_path(skel)
    result = {"skeleton": skel, "longest_path": longest_path}

    # __import__('pdb').set_trace()
    # result = dask.compute(result, scheduler="single-threaded")
    # __import__('pdb').set_trace()
    # np.save("/mmfs1/data/adhinart/dumb/welp.npy", result)
    # print("saved")

    return result