import numpy as np
import argparse
from cloudvolume import CloudVolume
from omegaconf import OmegaConf
from utils import DotDict


def viewer(conf):
    raw = CloudVolume(f"file://{conf.data.raw_layer}")
    spine = CloudVolume(f"file://{conf.data.spine_layer}")
    seg = CloudVolume(f"file://{conf.data.seg_layer}")
    raw_unique = raw.unique(bbox=np.s_[:, :, :])
    spine_unique = spine.unique(bbox=np.s_[:, :, :])
    seg_unique = seg.unique(bbox=np.s_[:, :, :])
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

    # cast to dictionary, because hash of OmegaConf fields depend on greater object
    conf = OmegaConf.to_container(conf, resolve=True)
    assert isinstance(conf, dict), "conf must be a dictionary"
    # allow dot access
    conf = DotDict(conf)

    viewer(conf)
