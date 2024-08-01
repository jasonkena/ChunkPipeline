import numpy as np
from cloudvolume import CloudVolume, Skeleton
import kimimaro
import socket
from collections import defaultdict

import argparse
from omegaconf import OmegaConf


def read_mappings(mapping: np.ndarray):
    """
    Nx2 array [N, (seg_id, trunk_id)

    Parameters
    ----------
    mapping
    """
    assert len(np.unique(mapping[:, 0])) == mapping.shape[0]
    assert 0 not in mapping[:, 0]

    seg_to_trunk = {int(k): int(v) for k, v in mapping}
    trunk_to_segs = defaultdict(list)
    for seg_id, trunk_id in mapping:
        trunk_to_segs[int(trunk_id)].append(int(seg_id))
    trunk_to_segs = dict(trunk_to_segs)

    return seg_to_trunk, trunk_to_segs


def main(conf):
    vol = CloudVolume(f"file://{conf.data.output_layer}")
    breakpoint()
    vol.viewer()
    breakpoint()
    mapping = np.load(conf.data.mapping)

    seg_to_trunk, trunk_to_segs = read_mappings(mapping)
    seg_ids = seg_to_trunk.keys()
    trunk_ids = trunk_to_segs.keys()

    # NOTE: some of these skeletons are empty
    skeletons = {int(k): vol.skeleton.get(k) for k in seg_ids}
    for seg_id in seg_to_trunk.keys():
        if len(skeletons[seg_id].vertices) == 0:
            print(f"ID: {seg_id} is empty")

    merged = {}
    for trunk_id in trunk_ids:
        merged[trunk_id] = Skeleton.simple_merge(
            [
                skeletons[seg_id]
                for seg_id in trunk_to_segs[trunk_id]
                if len(skeletons[seg_id].vertices) > 0
            ]
        )

    np.savez("merged.npz", skel=merged)


def local(conf):
    merged = np.load("merged.npz", allow_pickle=True)["skel"].item()
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        help="List of configuration files.",
        required=True,
    )

    args = parser.parse_args()
    print(args.config)

    confs = [OmegaConf.load(c) for c in args.config]
    conf = OmegaConf.merge(*confs)

    if "think-jason" in socket.gethostname():
        local(conf)
    else:
        main(conf)
