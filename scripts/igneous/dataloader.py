import numpy as np
import zarr
from utils import get_conf, groupby
from visualize import read_mappings
from scipy.spatial import KDTree
import networkx as nx

from cloudvolume import Skeleton
from typing import Optional, List, Set, Iterator, Tuple, Dict
from collections import defaultdict


def find_paths(
    G: nx.Graph,
    start: int,
    target_weight: float,
    path: Optional[List[int]] = None,
    seen: Optional[Set[int]] = None,
    current_weight: float = 0,
) -> Iterator[Tuple[int]]:
    """
    Given a trunk skeleton graph, find all (non-intesecting) paths from a starting node to a node with total weight (slightly) greater than a target weight

    Parameters
    ----------
    G
    start: index of the starting node
    target_weight

    path: variable used for recursion
    seen: variable used for recursion
    current_weight: variable used for recursion

    Returns
    -------
    Iterator[np.ndarray]:
    """
    if path is None:
        path = [start]
    if seen is None:
        seen = {start}

    # get direct descendants
    desc = nx.descendants_at_distance(G, start, 1)
    valid_desc = [n for n in desc if n not in seen]
    for n in valid_desc:
        edge_weight = G[start][n]["weight"]
        new_weight = current_weight + edge_weight
        if new_weight >= target_weight:
            yield tuple(path + [n])
        else:
            yield from find_paths(
                G, n, target_weight, path + [n], seen.union([n]), new_weight
            )


def get_random_path(G: nx.Graph, target_weight: float) -> Tuple[int]:
    """
    Given a trunk skeleton graph, find a random path greater than a target weight

    Parameters
    ----------
    G
    target_weight

    Returns
    -------
    Tuple[int]:
    """
    start = np.random.choice(list(G.nodes))
    paths = list(find_paths(G, start=start, target_weight=target_weight))
    return paths[np.random.choice(len(paths))]


def find_all_paths(G: nx.Graph, target_weight: float) -> List[Tuple[int]]:
    """
    Given a trunk skeleton graph, find all (non-intesecting) paths greater than a target weight

    Parameters
    ----------
    G
    target_weight

    Returns
    -------
    List[Tuple[int]]:
    """
    paths = []
    for node in G.nodes:
        paths.extend(list(find_paths(G, start=node, target_weight=target_weight)))
    # deduplicate paths (A -> B is the same as B -> A)
    canonical_paths = []
    for path in paths:
        if path[0] > path[-1]:
            canonical_paths.append(tuple(reversed(path)))
        else:
            canonical_paths.append(path)

    return list(set(canonical_paths))


def nx_from_skel(skel: Skeleton) -> nx.Graph:
    """
    Given a cloudvolume Skeleton, convert it into an undirected graph, with edge weights as the euclidean distance between vertices

    Parameters
    ----------
    skel
    """
    edges = []
    for a, b in skel.edges:
        l2 = np.linalg.norm(skel.vertices[a] - skel.vertices[b])
        edges.append((a, b, l2))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


def link_seeds(
    seed_data, seg_to_trunk, trunk_to_segs
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Returns a dictionary {seed_id : seed_id_of_trunk} and {seed_id_of_trunk: [seed_id of trunks and spines]}
    Entire spine segments are assigned to the id of the closest trunk vertex

    Parameters
    ----------
    seed_data
    seg_to_trunk
    """
    seg_ids = sorted(seg_to_trunk.keys())
    trunk_ids = sorted(trunk_to_segs.keys())
    spine_ids = sorted(list(set(seg_ids) - set(trunk_ids)))

    skeleton_ids, groups = groupby(seed_data, seed_data["skeleton_id"])
    skel_id_to_group = {skel_id: group for skel_id, group in zip(skeleton_ids, groups)}

    seed_id_to_trunk_seed_id = {}
    for skel_id, group in zip(skeleton_ids, groups):
        assert (skel_id in trunk_ids) ^ (skel_id in spine_ids)
        if skel_id in trunk_ids:
            for x in group["seed_id"]:
                seed_id_to_trunk_seed_id[x] = x
        else:
            trunk_group = skel_id_to_group[seg_to_trunk[skel_id]]
            # NOTE: everything here is isotropic
            trunk_pc = np.stack([trunk_group[f"seed_coord_{c}"] for c in "zyx"], axis=1)
            trunk_seed_id = trunk_group["seed_id"]
            spine_pc = np.stack([group[f"seed_coord_{c}"] for c in "zyx"], axis=1)
            spine_seed_id = group["seed_id"]

            dist, idx = get_closest(spine_pc, trunk_pc)
            closest_trunk_idx = trunk_seed_id[idx[np.argmin(dist)]]
            for x in spine_seed_id:
                seed_id_to_trunk_seed_id[x] = closest_trunk_idx

    trunk_seed_id_to_seed_ids = defaultdict(list)
    for k, v in seed_id_to_trunk_seed_id.items():
        trunk_seed_id_to_seed_ids[v].append(k)
    trunk_seed_id_to_seed_ids = dict(trunk_seed_id_to_seed_ids)

    return seed_id_to_trunk_seed_id, trunk_seed_id_to_seed_ids


def get_closest(pc_a, pc_b):
    """
    For each point in pc_a, find the closest point in pc_b
    Returns the distance and index of the closest point in pc_b for each point in pc_a
    Parameters
    ----------
    pc_a : [Mx3]
    pc_b : [Nx3]
    """
    tree = KDTree(pc_b)
    dist, idx = tree.query(pc_a, workers=-1)

    if np.max(idx) >= pc_b.shape[0]:
        raise ValueError("idx is out of range")

    return dist, idx


def skel_id_to_seed_id_mapping(seed_data):
    mapping = {}
    for row in seed_data:
        key = (row["skeleton_id"], row["vertex_id"])
        assert key not in mapping
        mapping[key] = row["seed_id"]
    return mapping


def seed_id_to_row_mapping(seed_data):
    mapping = {}
    for row in seed_data:
        key = row["seed_id"]
        assert key not in mapping
        mapping[key] = row
    return mapping


def weighted_random_sample(lengths: List[int], total_samples: int):
    """
    Determines the number of samples to take from each group based on the length of the group

    Parameters
    ----------
    lengths
    total_samples
    """
    weights = np.array(lengths) / sum(lengths)
    chosen_arrays = np.random.choice(
        len(lengths), size=total_samples, p=weights, replace=True
    )

    sample_counts = np.bincount(chosen_arrays, minlength=len(lengths))
    return sample_counts


def sample_path(pc_group, trunk_path, trunk_seed_id_to_seed_ids, num_points):
    # NOTE: limitation
    # since everything is done via feature transform, if the point cloud of another dendrite is closer to trunk of another, it will be included here
    seed_ids = []
    for seed_id in trunk_path:
        seed_ids.extend(trunk_seed_id_to_seed_ids[seed_id])
    lengths = [len(pc_group[str(seed_id)]) for seed_id in seed_ids]

    sample_counts = weighted_random_sample(lengths, num_points)
    points = []
    for i, seed_id in enumerate(seed_ids):
        points.extend(
            pc_group[str(seed_id)][
                np.random.choice(lengths[i], sample_counts[i], replace=True)
            ]
        )
    points = np.stack(points, axis=0)

    return points


def vertex_path_to_seed_path(trunk_id, vertex_path, seed_id_to_trunk_seed_id):
    return [
        seed_id_to_trunk_seed_id[(trunk_id, vertex_id)] for vertex_id in vertex_path
    ]


def visualize_batch(pc, trunk_skel):
    # assumes anisotropic
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(float))
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    pcd_trunk = o3d.geometry.PointCloud()
    pcd_trunk.points = o3d.utility.Vector3dVector(trunk_skel.astype(float))
    pcd_trunk.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd_trunk])


def main(conf):
    """
    seg_id: canonical labelling for each segment trunk/spines
    trunk_id: canonical labelling for each trunk (subset of seg_id)

    skeleton_ids: for each vertex, which seg_id
    vertex_ids: for each vertex, which index in the skeleton (multiple 0s)
    seed_coord_z/y/x: for each vertex, the ISOTROPIC coordinates
    seed_id: canonical vertex

    skeletons: cloudvolume skeletons {seg_id: skel}
    max_radius: max radius of all skeletons
    """
    if mp.is_remote:
        mapping = np.load(conf.data.mapping)
        seg_to_trunk, trunk_to_segs = read_mappings(mapping)

        seed = np.load(conf.data.seed, allow_pickle=True)
        seed_data, skeletons = seed["seed_data"], seed["skeletons"]

        seed_id_to_trunk_seed_id, trunk_seed_id_to_seed_ids = link_seeds(
            seed_data, seg_to_trunk, trunk_to_segs
        )
        skeleton_id_to_seed_id = skel_id_to_seed_id_mapping(seed_data)

        skeletons = skeletons.item()
        skeletons = {k: nx_from_skel(v) for k, v in skeletons.items()}
        trunk_ids = sorted(trunk_to_segs.keys())

        random_trunk_id = np.random.choice(trunk_ids)

        random_path = vertex_path_to_seed_path(
            random_trunk_id,
            get_random_path(skeletons[random_trunk_id], conf.dataloader.path_length),
            skeleton_id_to_seed_id,
        )

        # contains ANISOTROPIC coords
        pc_group = zarr.open_group(conf.data.pc_zarr, mode="r")

        seed_id_to_row = seed_id_to_row_mapping(seed_data)
        # NOTE: zyx are in anisotropic coords
        points = sample_path(
            pc_group, random_path, trunk_seed_id_to_seed_ids, conf.dataloader.num_points
        )

        trunk_points = [seed_id_to_row[seed_id] for seed_id in random_path]
        trunk_pc = np.stack(
            [
                np.array([point[f"seed_coord_{c}"] for c in "zyx"])
                for point in trunk_points
            ],
            axis=0,
        )

        pc = np.stack([points["z"], points["y"], points["x"]], axis=1) * np.array(
            conf.anisotropy
        )
        mp.save((pc, trunk_pc))
    else:
        pc, trunk_pc = mp.load()
        visualize_batch(pc, trunk_pc)


from magicpickle import MagicPickle

if __name__ == "__main__":
    np.random.seed(42)
    with MagicPickle("think-jason") as mp:
        if mp.is_remote:
            conf = get_conf()
            mp.save(conf)
            main(conf)
        else:
            conf = mp.load()
            main(conf)
