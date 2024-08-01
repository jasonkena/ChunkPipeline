import os
import argparse
from cloudvolume import Vec
from omegaconf import OmegaConf
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from utils import DotDict


def main(conf):
    tq = LocalTaskQueue(parallel=conf.n_jobs_mesh)
    layer = f"file://{conf.data.output_layer}"
    # create conf.data.output_layer/mesh directory
    if not os.path.exists(f"{conf.data.output_layer}/mesh"):
        os.makedirs(f"{conf.data.output_layer}/mesh")

    tasks = tc.create_meshing_tasks(  # First Pass
        layer, **conf.mesh  # Which data layer
    )
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(layer, **conf.mesh_merge)
    tq.insert(tasks)
    tq.execute()


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

    main(conf)
