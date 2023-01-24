from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_load_h5, task_foundation_seg


class FoundationPipeline(Pipeline):
    def run(self):
        # load main low res datasets
        h5 = self.add(task_load_h5, "h5", cfg_groups=["GENERAL", "H5"])
        seg = self.add(
            task_foundation_seg,
            "seg",
            cfg_groups=["GENERAL", "FOUNDATION"],
            depends_on=[h5],
        )

        self.compute()
        self.export(
            "foundation.zip",
            ["seg/seg"],
            ["seg"],
        )
