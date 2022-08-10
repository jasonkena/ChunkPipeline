import sys
import shutil
from abc import ABC, abstractmethod

import numpy as np

import dask
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask_memusage import install
from dask.distributed import Client, LocalCluster, wait

from chunk_pipeline.utils import object_array
from chunk_pipeline.configs import Config

import zarr
import numcodecs


# https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
def flatten(l):
    return [item for sublist in l for item in sublist]


def to_tuple(l):
    return tuple(tuple(sublist) for sublist in l)


def parallelize(func):
    def wrapper(self, *args, **kwargs):
        slurm = self.cfg["SLURM"]
        misc = self.cfg["MISC"]

        if (slurm_exists := shutil.which("sbatch")) is None:
            print("SLURM not available, falling back to LocalCluster")

        dask.config.set({"distributed.comm.timeouts.connect": "300s"})
        # dask.config.set({"distributed.comm.retry.count": 3})
        # dask.config.set({'distributed.scheduler.idle-timeout' : "5 minutes"})
        with SLURMCluster(
            local_directory=slurm["LOCAL_DIRECTORY"],
            job_name=slurm["PROJECT_NAME"],
            queue=slurm["PARTITIONS"],
            cores=slurm["CORES_PER_JOB"],
            memory=f"{slurm['MEMORY_PER_JOB']}GiB",
            scheduler_options={"dashboard_address": f":{slurm['DASHBOARD_PORT']}"},
            walltime=slurm["WALLTIME"],
            processes=slurm["NUM_PROCESSES_PER_JOB"],
            interface=slurm["INTERFACE"],
        ) if slurm_exists else LocalCluster() as cluster, Client(cluster) as client:

            print(cluster.dashboard_link)
            if slurm_exists:
                print(
                    "Asking for {} cores, {} GiB of memory, {} processes per job on SLURM".format(
                        slurm["CORES_PER_JOB"],
                        slurm["MEMORY_PER_JOB"],
                        slurm["NUM_PROCESSES_PER_JOB"],
                    )
                )
                cluster.scale(jobs=slurm["MIN_JOBS"])
            install(cluster.scheduler, misc["MEMUSAGE_PATH"])
            return func(self, *args, **kwargs)

    return wrapper


def iterdict(cfg, base_cfg=None):
    if base_cfg is None:
        base_cfg = cfg
    for key, value in cfg.items():
        if isinstance(value, dict):
            iterdict(value, base_cfg)
        elif isinstance(value, str):
            cfg[key] = value.replace("{TASK}", base_cfg["TASK"])


class Pipeline(ABC):
    def __init__(self, cfg):
        self._uid = -1
        self.tasks = {}

        iterdict(cfg)  # replace {TASK} with cfg["TASK"]
        self.cfg = cfg

        self.store = zarr.DirectoryStore(cfg["MISC"]["ZARR_PATH"])

        base_group = zarr.group(store=self.store)
        if "_config" in base_group:
            if cfg != base_group["_config"][0]:
                print("Config has changed, overwriting old config")

        base_group.create_dataset(
            "_config",
            data=object_array([self.cfg]),
            object_codec=numcodecs.Pickle(),
            overwrite=True,
        )

    def add(self, func, path, cfg_groups=[], depends_on=[], checkpoint=True):
        # path here is the zarr groups path
        self._uid += 1

        # what is used to compute whether to update attr/array
        # existing cfg_groups, with the implicit groups of dependencies
        implicit_cfg_groups = cfg_groups + flatten(
            [self.tasks[i]["implicit_cfg_groups"] for i in depends_on]
        )
        implicit_cfg_groups = sorted(list(set(implicit_cfg_groups)))
        implicit_depends_on = depends_on + flatten(
            [self.tasks[i]["implicit_depends_on"] for i in depends_on]
        )
        implicit_depends_on = sorted(list(set(implicit_depends_on)))

        result = {
            "func": func,
            "path": path,
            "cfg_groups": cfg_groups,
            "depends_on": depends_on,
            "checkpoint": checkpoint,
            "implicit_cfg_groups": implicit_cfg_groups,
            "implicit_depends_on": implicit_depends_on,
        }
        self.tasks[self._uid] = result

        return self._uid

    def wrap(self, task, *args, **kwargs):
        # _attrs contains any attributes that are not _Dask_ arrays
        # numpy arrays are passed into _attrs

        # what is passed to the func
        inner_cfg = {group: self.cfg[group] for group in task["cfg_groups"]}

        # what is used to check whether to update attr/array
        implicit_cfg = {group: self.cfg[group] for group in task["implicit_cfg_groups"]}
        group = zarr.open_group(store=self.store, path=task["path"], mode="a")

        # assuming that if _implicit_cfg is non empty, the other fields have been written to successfully
        if ("_implicit_cfg" in group) and (implicit_cfg == group["_implicit_cfg"][0]):
            result = group["_attrs"][
                0
            ].copy()  # not using group.attrs since pickling is not supported
            for key in group.keys():
                if key[0] != "_":
                    chunks = result.pop(f"_{key}")
                    result[key] = da.from_zarr(group[key], chunks=chunks)
            return False, result

        else:
            if "_implicit_cfg" in group:
                print(f"implicit_cfg has changed, will clear {group.path}")
            result = task["func"](inner_cfg, *args, **kwargs)
            assert isinstance(result, dict)
            return task["checkpoint"], result

    @parallelize
    def compute(self, task_uids=None):
        # if task_uids is None, compute all tasks
        if task_uids is None:
            task_uids = list(self.tasks.keys())
        # already includes reference to self task_uid
        implicit_depends_on = (
            flatten([self.tasks[i]["implicit_depends_on"] for i in task_uids])
            + task_uids
        )
        implicit_depends_on = sorted(list(set(implicit_depends_on)))

        results = {}
        needs_checkpoint = []

        # compute everything else
        for i in implicit_depends_on:
            # reference results of previous tasks
            dependencies = [results[j] for j in self.tasks[i]["depends_on"]]
            assert all(isinstance(dependency, dict) for dependency in dependencies)

            # get either a dict or a function back
            checkpoint, results[i] = self.wrap(self.tasks[i], *dependencies)
            if checkpoint:
                needs_checkpoint.append(i)

        self.checkpoint(results, needs_checkpoint)

        print("Computed all tasks")
        return results

    def checkpoint(self, results, idx):
        results = {i: results[i].copy() for i in idx}
        pointer = dask.persist([results[i] for i in idx])
        groups = [
            zarr.open_group(store=self.store, path=self.tasks[i]["path"], mode="w")
            for i in idx
        ]
        stored = []

        # checkpoint each array
        for i in idx:
            for key in list(results[i].keys()):
                value = results[i].pop(key)
                if isinstance(value, da.Array):
                    if np.isnan(value.shape).any():
                        value.compute_chunk_sizes()
                    # chunks will be saved in _attrs
                    results[i][f"_{key}"] = value.chunks

                    stored.append(
                        da.to_zarr(
                            value,
                            url=self.store.path,
                            component=f"{groups[i].path}/{key}",
                            compute=False,
                        )
                    )
        _, attrs = dask.compute(stored, results)
        for i in idx:
            groups[i].create_dataset(
                "_attrs",
                data=object_array([attrs[i]]),
                object_codec=numcodecs.Pickle(),
            )
            implicit_cfg = {
                group: self.cfg[group] for group in self.tasks[i]["implicit_cfg_groups"]
            }
            groups[i].create_dataset(
                "_implicit_cfg",
                data=object_array([implicit_cfg]),
                object_codec=numcodecs.Pickle(),
            )
        return

    @abstractmethod
    def run(self):
        """
        one = self.add(func1, path, cfg_groups=[], depends_on=[], checkpoint=True)
        two = self.add(func, path, cfg_groups=[], depends_on=[], checkpoint=True)
        three = self.add(func, path, cfg_groups=[], depends_on=[], checkpoint=True)
        self.compute([one, two, three])
            or
        self.compute() to compute all tasks
        """
        return
