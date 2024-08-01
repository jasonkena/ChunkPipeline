import numpy as np
import argparse
from omegaconf import OmegaConf
from joblib import Parallel, delayed
import h5py
from to_precomputed import get_chunks
from tqdm import tqdm

"""
just used to debug seg_den_seg
for some reason the ids are between 0-255, despite the dtype being uint16
there are like 300 unique segments in the mapping returned by to_precomputed output_layer for mouse and human
"""


def get_unique_chunk(chunk, seg, seg_key):
    seg_file = h5py.File(seg, "r")[seg_key]
    res = np.unique(seg_file[chunk])
    print(res)

    return res


def main(conf):
    seg_file = h5py.File(conf.data["seg"], "r")[conf.data["seg_key"]]
    chunks = get_chunks(tuple(seg_file.shape) + (1,), tuple(seg_file.chunks))
    res = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs, return_as="generator")(
                delayed(get_unique_chunk)(
                    c,
                    conf.data["seg"],
                    conf.data["seg_key"],
                )
                for c in chunks
            ),
            total=len(chunks),
            leave=False,
        )
    )
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

    main(conf)
