import zarr

from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_load_h5, task_bbox, task_extract_seg, task_skeletonize


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
                    depends_on=[bbox, h5],
                )
            )
        skeletons = []
        for i in range(n):
            skeletons.append(
                self.add(
                    task_skeletonize(i),
                    f"skeleton_{i}",
                    cfg_groups=["GENERAL", "KIMI"],
                    depends_on=[bbox, extracted[i]],
                )
            )

        self.compute()
        sources = []
        dests = []
        for i in range(n):
            sources.append(f"skeleton_{i}/skeleton")
            sources.append(f"skeleton_{i}/longest_path")
            dests.append(f"{i}/skeleton")
            dests.append(f"{i}/longest_path")

        self.export("skel.h5", sources, dests)
