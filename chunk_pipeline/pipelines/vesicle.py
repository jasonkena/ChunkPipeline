from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_run_vesicle


class VesiclePipeline(Pipeline):
    def run(self):
        # load main low res datasets
        vesicle = self.add(
            task_run_vesicle, "vesicle", cfg_groups=["GENERAL", "VESICLE"]
        )

        self.compute()
