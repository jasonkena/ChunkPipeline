from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_generate_original, task_coarse_segment


class CoarsePipeline(Pipeline):
    def run(self):
        original = self.add(
            task_generate_original,
            "original",
            cfg_groups=["GENERAL", "COARSE_ORIGINAL"],
        )
        seg = self.add(
            task_coarse_segment,
            "coarse_seg",
            cfg_groups=["GENERAL", "COARSE"],
            depends_on=[original],
        )
        self.compute()
