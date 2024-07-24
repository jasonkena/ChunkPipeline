import zarr

from chunk_pipeline.pipelines import Pipeline
from chunk_pipeline.tasks import (
    task_load_h5,
    task_bbox,
    task_extract_seg,
    task_skeletonize,
    task_generate_point_cloud,
    task_generate_point_cloud_segments,
    task_generate_l1_from_pc,
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

        # NOTE: kimimaro skeletons are broken
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
        print("done computing kimimaro skeletons")

        point_clouds = []
        for i in range(n):
            # for i in range(34, 47):
            point_clouds.append(
                self.add(
                    task_generate_point_cloud,
                    f"point_cloud_{i}",
                    cfg_groups=["GENERAL", "PC"],
                    depends_on=[extracted[i]],
                )
            )
            if i % 5 == 0:
                print("Computing point cloud", i)
                self.compute()
        self.compute()
        print("done computing point clouds")
        #
        l1 = []
        for i in range(n):
            l1.append(
                self.add(
                    task_generate_l1_from_pc,
                    f"l1_{i}",
                    cfg_groups=["GENERAL", "L1"],
                    depends_on=[point_clouds[i]],
                )
            )
            if i % 10 == 0 and i != 0:
                print("Computing l1", i)
                self.compute()
        self.compute()
        print("done computing l1")

        point_cloud_segments = []
        for i in range(n):
            point_cloud_segments.append(
                self.add(
                    task_generate_point_cloud_segments,
                    f"point_cloud_segments_{i}",
                    cfg_groups=["GENERAL", "FRENET"],
                    depends_on=[point_clouds[i], l1[i]],
                    # depends_on=[point_clouds[i], skeletons[i]],
                )
            )
            if i % 10 == 0 and i != 0:
                print("Computing point cloud segments", i)
                # this already uses ~ 500 GB of RAM
                self.compute()
        self.compute()
        print("done computing pre-export")
        # return

        base_path = self.cfg["MISC"]["ZARR_PATH"]
        task = self.cfg["TASK"]

        
        import numpy as np
        import glob
        import os
        from tqdm import tqdm

        save_path = os.path.join(base_path, f"pc_export_{task}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in tqdm(range(n)):
            pc_group = zarr.open_group(f"{base_path}/point_cloud_{i}")
            pc_seg_group = zarr.open_group(f"{base_path}/point_cloud_segments_{i}")
            idx = pc_group["idx"][:]
            spine = pc_group["spine"][:]
            seg = pc_group["seg"][:]
            centerline = pc_seg_group["_attrs"][0]["reference"]

            np.savez(os.path.join(save_path, f"pc_{i}.npz"), idx=idx, spine=spine, seg=seg, centerline=centerline)
            keys = list(pc_seg_group.keys())
            ids = [key.split("_")[-1] for key in keys]
            ids = [int(x) for x in ids if x.isdigit()]
            ids = sorted(set(ids))

            # for each slice
            for j in tqdm(ids,leave=False):
                pc = pc_seg_group[f"pc_{j}"][:]
                closest_idx = pc_seg_group[f"closest_idx_{j}"][:]
                np.savez(os.path.join(save_path, f"pc_seg_{i}_{j}.npz"), pc=pc, closest_idx=closest_idx)

        print("saving done")

        # see git history for how to use self.export
