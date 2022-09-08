import zarr

from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import (
    task_load_h5,
    task_bbox,
    task_extract_seg,
    task_skeletonize,
    task_generate_point_cloud,
)


class DendritePipeline(Pipeline):
    def run(self):
        # load raw, seg, spine h5s
        h5 = self.add(task_load_h5, "h5", cfg_groups=["GENERAL", "H5"])

        # compute bbox from raw
        bbox = self.add(
            task_bbox("raw"),
            "bbox",
            cfg_groups=["GENERAL"],
            depends_on=[h5],
        )
        self.compute([bbox])
        _bbox = zarr.load(store=self.store, path="bbox/bbox")
        n = _bbox.shape[0]

        # compute dendrite specific chunks
        extracted = []
        for i in range(n):
            extracted.append(
                self.add(
                    task_extract_seg(i),
                    f"extracted_{i}",
                    cfg_groups=["GENERAL"],
                    depends_on=[bbox, h5],
                )
            )
        self.compute()
        skeletons = []
        for i in range(n):
            skeletons.append(
                self.add(
                    task_skeletonize,
                    f"skeleton_{i}",
                    cfg_groups=["GENERAL", "KIMI"],
                    depends_on=[extracted[i]],
                )
            )
        self.compute()

        point_clouds = []
        # for i in [0]:
        # for i in [34]:
        # for i in [13]:
        # for i in [22]:
        for i in range(n):
            point_clouds.append(
                self.add(
                    task_generate_point_cloud,
                    f"point_cloud_{i}",
                    cfg_groups=["GENERAL", "PC"],
                    depends_on=[extracted[i], skeletons[i]],
                )
            )

        self.compute()

        # export skeletons
        skels = []
        longest_paths = []
        for i in range(n):
            skels.append(self.load(f"skeleton_{i}/skeleton"))
            longest_paths.append(self.load(f"skeleton_{i}/longest_path"))
        from chunk_pipeline.utils import object_array

        skels = object_array(skels)
        longest_paths = object_array(longest_paths)

        self.export("skel.zip", [skels, longest_paths], ["skel", "longest_path"])
