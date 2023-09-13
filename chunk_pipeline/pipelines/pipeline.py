import os
import shutil
from abc import ABC, abstractmethod
import itertools
import numcodecs

import pdb

import numpy as np

import dask
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster, wait
from distributed.diagnostics.plugin import WorkerPlugin
from dask.graph_manipulation import bind

import chunk_pipeline.tasks.chunk as chunk
from chunk_pipeline.utils import object_array
from chunk_pipeline.configs import Config

import zarr
import numcodecs

import logging
from dask_memusage import install


# https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
def flatten(l):
    return [item for sublist in l for item in sublist]


def to_tuple(l):
    return tuple(tuple(sublist) for sublist in l)


#
# def parallelize(func):
#     def wrapper(self, *args, **kwargs):
#                 # cluster.scale(jobs=slurm["MIN_JOBS"])
#             return func(self, *args, **kwargs)
#
#     return wrapper
#


def iterdict(cfg, base_cfg=None):
    if base_cfg is None:
        base_cfg = cfg
    for key, value in cfg.items():
        if isinstance(value, dict):
            iterdict(value, base_cfg)
        elif isinstance(value, str):
            cfg[key] = value.replace("{TASK}", base_cfg["TASK"])


class ResolveWorker(WorkerPlugin):
    def setup(self, worker):
        self.worker = worker


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
                logging.info("Config has changed, overwriting old config")

        self.zarr_kwargs = {
            "compressor": numcodecs.Zstd(),  # use zstd as it has no block size limit
            "object_codec": numcodecs.Pickle(),
        }
        base_group.create_dataset(
            "_config", data=object_array([self.cfg]), overwrite=True, **self.zarr_kwargs
        )

        slurm = self.cfg["SLURM"]
        misc = self.cfg["MISC"]

        if (slurm_exists := shutil.which("sbatch")) is None:
            logging.info("SLURM not available, falling back to LocalCluster")

        self.cluster = (
            SLURMCluster(
                job_name=slurm["PROJECT_NAME"],
                queue=slurm["PARTITIONS"],
                cores=slurm["CORES_PER_JOB"],
                # --mem=0 broken for some reason, added header_skip, see https://github.com/dask/dask-jobqueue/issues/497
                job_mem="0GB",
                memory=0,
                # memory="0GB",  # with --exclusive, this allocates all memory on node
                # memory=f"{slurm['MEMORY_PER_JOB']}GiB",
                scheduler_options={
                    "dashboard_address": f":{slurm['DASHBOARD_PORT']}",
                    "interface": "ib0",
                },
                walltime=f"{slurm['WALLTIME']}:00:00",
                processes=slurm["NUM_PROCESSES_PER_JOB"],
                # interface=slurm["INTERFACE"],
                worker_extra_args=[
                    "--interface",
                    slurm["INTERFACE"],
                    "--nthreads",
                    str(
                        n_threads := int(
                            slurm["MEMORY_PER_JOB"]
                            / (
                                slurm["NUM_PROCESSES_PER_JOB"]
                                * slurm["MEMORY_PER_TASK"]
                            )
                        )
                    ),
                    # auto compute memory
                    # f'--memory-limit="'
                    # + str(int(slurm["MEMORY_PER_JOB"] / slurm["NUM_PROCESSES_PER_JOB"]))
                    # + 'GiB"',
                    "--local-directory",
                    slurm["LOCAL_DIRECTORY"],
                    "--lifetime",
                    f"{slurm['WALLTIME']*60-5}m",
                    "--lifetime-stagger",
                    "4m",
                    # f"""--memory-limit=$(grep MemFree /proc/meminfo | awk '{{print $2, "/ 1000000 "}}' | bc)GiB""",
                    f"""--memory-limit=$(grep MemFree /proc/meminfo | awk '{{print $2, "/ 1000000 / {slurm["NUM_PROCESSES_PER_JOB"]} "}}' | bc)GiB""",
                ],
                log_directory=slurm["LOG_DIRECTORY"],
                # allocate all CPUs and run only one job per node
                job_extra_directives=["--exclusive", "--gpus-per-node=0"],
            )
            if slurm_exists and not misc["ENABLE_MEMUSAGE"]
            else LocalCluster(
                **(
                    {"threads_per_worker": 1, "n_workers": 5, "memory_limit": "12GiB"}
                    if misc["ENABLE_MEMUSAGE"]
                    else {}
                )
            )
        )
        if misc["ENABLE_MEMUSAGE"]:
            install(self.cluster.scheduler, misc["MEMUSAGE_PATH"])

        self.client = Client(self.cluster)

        logging.info(self.cluster.dashboard_link)
        # print(self.cluster.job_script())
        if slurm_exists:
            logging.info(
                "Asking for {} cores, {} GiB of memory, {} processes per job on SLURM, {} threads per process".format(
                    slurm["CORES_PER_JOB"],
                    slurm["MEMORY_PER_JOB"],
                    slurm["NUM_PROCESSES_PER_JOB"],
                    n_threads,
                )
            )
            self.cluster.adapt(
                minimum=slurm["MIN_JOBS"] * slurm["NUM_PROCESSES_PER_JOB"],
                maximum=slurm["MAX_JOBS"] * slurm["NUM_PROCESSES_PER_JOB"],
                worker_key=lambda state: state.address.split(":")[0],
                interval=slurm["ADAPT_INTERVAL"],
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
                        logging.info(f"chunks missing in {key}, assuming object_array")
                        chunks = None
                    result[key] = da.from_zarr(group[key], chunks=chunks)
            return False, result

        else:
            if "_implicit_cfg" in group:
                logging.info(f"implicit_cfg has changed, will clear {group.path}")
            result = task["func"](inner_cfg, *args, **kwargs)
            assert isinstance(result, dict)
            return task["checkpoint"], result

    def debug_compute(self, *args, **kwargs):
        # if any(args) or len(kwargs):
        #     for x in args[0]:
        #         x.visualize(filename=f"{x.key}.svg")
        #         print("visualizing")
        #     __import__('pdb').set_trace()
        futures = self.client.compute(args, **kwargs)
        results = []
        wait(futures)
        for future in futures:
            if future.status == "error":
                logging.error(repr(future.exception()))
                logging.error("Error raised, dropping into pdb")
                pdb.runcall(self.client.recreate_error_locally, future)
                raise future.exception()
            else:
                results.append(future.result())
        if len(results) != len(args):
            breakpoint()
        return results
        # return dask.compute(*args)

    # @parallelize
    def compute(self, task_uids=None):
        for dataset in self.client.list_datasets():
            self.client.unpublish_dataset(dataset)

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

        logging.info("Computed all tasks")
        return results

    def checkpoint(self, original_results, idx):
        results = {i: original_results[i].copy() for i in idx}
        groups = {
            i: zarr.open_group(store=self.store, path=self.tasks[i]["path"], mode="w")
            for i in idx
        }
        stored = []
        object_array_paths = []

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
                        logging.warning(f"{key} has irregular chunks; please rechunk")
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
                            # overwrite=True,
                            **self.zarr_kwargs,
                        )
                    )
                    if is_object:
                        object_array_paths.append(path)

        _, attrs = self.debug_compute(stored, results)
        self.fix_object_arrays(object_array_paths)

        # fix object arrays

        for i in idx:
            groups[i].create_dataset(
                "_attrs", data=object_array([attrs[i]]), **self.zarr_kwargs
            )
            implicit_cfg = {
                group: self.cfg[group] for group in self.tasks[i]["implicit_cfg_groups"]
            }
            groups[i].create_dataset(
                "_implicit_cfg", data=object_array([implicit_cfg]), **self.zarr_kwargs
            )
        return

    def fix_object_arrays(self, paths):
        dirs, names = [], []
        shapes, dtypes, arrays = [], [], []
        for path in paths:
            tokens = path.split("/")
            dir, name = "/".join(tokens[:-1]), tokens[-1]
            assert name[0] == "_"
            dirs.append(dir)
            names.append(name)

            array = da.from_zarr(self.load(path))
            array = array.rechunk([1] * len(array.shape))
            shape, dtype = chunk.chunk(
                lambda x: [x.item().shape, x.item().dtype],
                [array],
                output_dataset_dtypes=[object, object],
            )
            shapes.append(shape)
            dtypes.append(dtype)
            arrays.append(array)
        shapes, dtypes = self.debug_compute(shapes, dtypes)

        for i in range(len(paths)):
            dtype = set(dtypes[i].flatten().tolist())
            try:
                assert len(dtype) == 1
            except:
                __import__("pdb").set_trace()
            dtypes[i] = dtype.pop()

        stored = []
        for i in range(len(paths)):
            chunks = [[np.nan] * x for x in arrays[i].shape]
            for chunk_idx in itertools.product(*[range(len(x)) for x in chunks]):
                for dim, dim_idx in enumerate(chunk_idx):
                    computed_shape = shapes[i][chunk_idx][dim]
                    if np.isnan(chunks[dim][dim_idx]):
                        chunks[dim][dim_idx] = computed_shape
                    else:
                        # asserts that chunk size matches known values
                        assert chunks[dim][dim_idx] == computed_shape
            chunks = tuple([tuple(x) for x in chunks])
            reconstructed = da.map_blocks(
                lambda x: x.item(),
                arrays[i],
                dtype=dtypes[i],
                chunks=chunks,
                name="reconstruct_unknown",
            ).rechunk()
            # NOTE: object_arrays will have no chunks attr

            stored.append(
                da.to_zarr(
                    reconstructed,
                    url=self.store.path,
                    component=os.path.join(dirs[i], names[i][1:]),
                    compute=False,
                    # overwrite=True,
                )
            )
        return self.debug_compute(stored)

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
                group.create_dataset(name, data=source, **self.zarr_kwargs)
        logging.info("Export finished")

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
