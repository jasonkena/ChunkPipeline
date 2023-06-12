from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_load_h5, task_generate_snemi_l1_from_vol


class SnemiL1Pipeline(Pipeline):
    def run(self):
        # load main dataset
        h5 = self.add(task_load_h5, "h5", cfg_groups=["GENERAL", "H5"])

        l1 = self.add(
            task_generate_snemi_l1_from_vol,
            "l1",
            cfg_groups=["GENERAL", "L1"],
            depends_on=[h5],
        )

        self.compute()
        self.export("l1.zip", ["l1/_attrs"], ["l1"])
