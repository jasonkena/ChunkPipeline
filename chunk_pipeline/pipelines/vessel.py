from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import task_load_h5, task_clean_sam


class VesselPipeline(Pipeline):
    def run(self):
        # load main
        h5 = self.add(task_load_h5, "h5", cfg_groups=["GENERAL", "H5"])
        self.compute()

        cleaned_sam = self.add(
            task_clean_sam,
            "cleaned_sam",
            cfg_groups=["GENERAL", "VESSEL"],
            depends_on=[h5],
        )
        self.compute()
        self.export(
            "cleaned_sam.zip",
            ["cleaned_sam/seg", "cleaned_sam/voxel_counts"],
            ["seg", "voxel_counts"],
        )
