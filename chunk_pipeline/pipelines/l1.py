from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_generate_l1_from_npz


class L1Pipeline(Pipeline):
    def run(self):
        # load main low res datasets
        l1 = self.add(
            task_generate_l1_from_npz,
            "l1",
            cfg_groups=["GENERAL", "NPZ", "L1"],
        )

        self.compute()
        self.export("l1.zip", ["l1/_attrs"], ["l1"])
