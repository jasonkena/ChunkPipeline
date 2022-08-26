import os
import shutil
from abc import ABC, abstractmethod
import itertools

import numpy as np

import dask
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster, wait
from dask.graph_manipulation import bind

import chunk_pipeline.tasks.chunk as chunk
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

        # # dask.config.set(scheduler='single-threaded')
        # from dask_memusage import install
        # cluster = LocalCluster(n_workers=10, threads_per_worker=1,
        #                memory_limit=None)
        # install(cluster.scheduler, misc["MEMUSAGE_PATH"])  # <-- INSTALL
        # client = Client(cluster)
        # print(cluster.dashboard_link)
        # return func(self, *args, **kwargs)
        #
        if (slurm_exists := shutil.which("sbatch")) is None:
            print("SLURM not available, falling back to LocalCluster")

        with dask.config.set(
            self.cfg["DASK_CONFIG"], scheduler="processes"
        ), SLURMCluster(
            job_name=slurm["PROJECT_NAME"],
            queue=slurm["PARTITIONS"],
            cores=slurm["CORES_PER_JOB"],
            memory=f"{slurm['MEMORY_PER_JOB']}GiB",
            scheduler_options={"dashboard_address": f":{slurm['DASHBOARD_PORT']}"},
            walltime=slurm["WALLTIME"],
            interface=slurm["INTERFACE"],
            worker_extra_args=[
                "--nworkers",
                str(slurm["NUM_PROCESSES_PER_JOB"]),
                "--nthreads",
                "1",
                f'--memory-limit="'
                + str(int(slurm["MEMORY_PER_JOB"] / slurm["NUM_PROCESSES_PER_JOB"]))
                + 'GiB"',
                "--local-directory",
                slurm["LOCAL_DIRECTORY"],
            ],
        ) if slurm_exists else LocalCluster() as cluster, Client(
            cluster
        ) as client:

            print(cluster.dashboard_link)
            # print(cluster.job_script())
            if slurm_exists:
                print(
                    "Asking for {} cores, {} GiB of memory, {} processes per job on SLURM".format(
                        slurm["CORES_PER_JOB"],
                        slurm["MEMORY_PER_JOB"],
                        slurm["NUM_PROCESSES_PER_JOB"],
                    )
                )
                cluster.scale(jobs=slurm["MIN_JOBS"])
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
                    if f"_{key}" in result:
                        chunks = result.pop(f"_{key}")
                    else:
                        print(f"chunks missing in {key}, assuming object_array")
                        chunks = None
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

    def checkpoint(self, original_results, idx):
        results = {i: original_results[i].copy() for i in idx}
        groups = {
            i: zarr.open_group(store=self.store, path=self.tasks[i]["path"], mode="w")
            for i in idx
        }
        stored = []

        # checkpoint each array
        for i in idx:
            for key in list(results[i].keys()):
                if isinstance(results[i][key], da.Array):
                    is_object = False
                    value = results[i].pop(
                        key
                    )  # remove arrays from the graph to form an attr graph
                    if np.isnan(value.shape).any():
                        value = chunk.chunk(lambda x: [x], [value], [object])
                        is_object = True
                    if not da.core._check_regular_chunks(value.chunks):
                        print(f"{key} has irregular chunks; please rechunk")
                        value = da.rechunk(
                            value,
                            chunks=(self.cfg["GENERAL"]["CHUNK_SIZE"][0],)
                            * len(value.shape),
                        )

                    # chunks will be saved in _attrs
                    if not is_object:
                        results[i][f"_{key}"] = value.chunks

                    path = f"{groups[i].path}/{'_'+key if is_object else key}"
                    stored.append(
                        da.to_zarr(
                            value,
                            url=self.store.path,
                            component=path,
                            compute=False,
                            object_codec=numcodecs.Pickle(),
                        )
                    )
                    if is_object:
                        stored.append(self.fix_object_array(path, stored[-1]))
        _, attrs = dask.compute(stored, results)

        # fix object arrays

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

    @dask.delayed
    def fix_object_array(self, path, old_store=None):
        tokens = path.split("/")
        dir, name = "/".join(tokens[:-1]), tokens[-1]
        assert name[0] == "_"

        array = da.from_zarr(self.load(path))
        array = array.rechunk([1] * len(array.shape))
        shape, dtype = chunk.chunk(
            lambda x: [x.item().shape, x.item().dtype],
            [array],
            output_dataset_dtypes=[object, object],
        )
        shape, dtype = dask.compute(shape, dtype)
        dtype = set(dtype.flatten().tolist())
        assert len(dtype) == 1
        dtype = dtype.pop()

        chunks = [[np.nan] * x for x in array.shape]
        for chunk_idx in itertools.product(*[range(len(x)) for x in chunks]):
            for dim, dim_idx in enumerate(chunk_idx):
                computed_shape = shape[chunk_idx][dim]
                if np.isnan(chunks[dim][dim_idx]):
                    chunks[dim][dim_idx] = computed_shape
                else:
                    # asserts that chunk size matches known values
                    assert chunks[dim][dim_idx] == computed_shape
        chunks = tuple([tuple(x) for x in chunks])
        reconstructed = da.map_blocks(
            lambda x: x.item(),
            array,
            dtype=dtype,
            chunks=chunks,
            name="reconstruct_unknown",
        ).rechunk()  # normalize chunks
        # NOTE: object_arrays will have no chunks attr

        return da.to_zarr(
            reconstructed,
            url=self.store.path,
            component=os.path.join(dir, name[1:]),
            compute=True,
        )

    def load(self, source):
        tokens = source.split("/")
        dir, name = "/".join(tokens[:-1]), tokens[-1]

        group = zarr.open_group(store=self.store, path=dir, mode="r")
        if name in group:
            return group[name]
        else:
            return group["_attrs"][0][name]

    def export(self, path, sources, dests):
        # path of h5 file to export to
        # sources is path to zarr arrays (with special handling for _attrs)
        # dests is path to h5 arrays

        # fail if exists
        assert len(sources) == len(dests)
        path = os.path.join(self.store.path, path)
        if os.path.exists(path):
            raise Exception("File already exists")
        file = zarr.group(zarr.storage.ZipStore(path))

        for i in range(len(sources)):
            if isinstance(sources[i], str):
                source = self.load(sources[i])
            else:
                source = sources[i]

            tokens = dests[i].split("/")
            dir, name = "/".join(tokens[:-1]), tokens[-1]
            if dir == "":
                dir = "/"
            group = file.require_group(dir)

            if isinstance(sources[i], str):
                zarr.copy(source, group, name)
            else:
                group.create_dataset(name, data=source, object_codec=numcodecs.Pickle())
        print("Exported finished")

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
