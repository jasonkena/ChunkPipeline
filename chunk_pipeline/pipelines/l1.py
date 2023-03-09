from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_load_nib, task_generate_l1_from_vol


class L1Pipeline(Pipeline):
    def run(self):
        # load main low res datasets
        nib = self.add(task_load_nib, "nib", cfg_groups=["GENERAL", "NIB"])
        l1 = self.add(
            task_generate_l1_from_vol,
            "l1",
            cfg_groups=["GENERAL", "L1"],
            depends_on=[nib],
        )

        self.compute()
        self.export("l1.zip", ["l1/_attrs"], ["l1"])
