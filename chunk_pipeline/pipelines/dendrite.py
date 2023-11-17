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
            if i % 10 == 0:
                print("Computing l1", i)
                self.compute()
        self.compute()
        print("done computing l1")
        __import__("pdb").set_trace()

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
            if i % 10 == 0:
                # if i % 10 == 0:
                print("Computing point cloud segments", i)
                # this already uses ~ 500 GB of RAM
                self.compute()
        self.compute()
        print("done computing pre-export")

        # export skeletons
        skels = []
        longest_paths = []

        # export point clouds
        idxs = []
        spines = []
        seg = []
        #
        # segments = []
        # segments_skel = []
        # segments_skel_gnb = []
        #
        import numpy as np
        import os

        # task_name = "seg_den"
        # base_path = f"/mmfs1/data/adhinart/dendrite/data/{task_name}/pc_export"
        # if not os.path.exists(base_path):
        #     os.makedirs(base_path)
        # for i in range(n):
        #     group = zarr.open_group(f"/mmfs1/data/adhinart/dendrite/data/{task_name}/point_cloud_segments_{i}")
        #     centerline = self.load(f"point_cloud_segments_{i}/centerline")
        #     np.savez(os.path.join(base_path,f"{task_name}_{i}_centerline.npz"), centerline)
        #     for j in range(6):
        #         pc = group[str(j)][:]
        #         np.savez(os.path.join(base_path,f"{task_name}_{i}_{j}.npz"), pc)
        # print("saving done")
        #

        for i in range(n):
            # skels.append(self.load(f"skeleton_{i}/skeleton"))
            # longest_paths.append(self.load(f"skeleton_{i}/longest_path"))
            idxs.append(f"point_cloud_{i}/idx")
            spines.append(f"point_cloud_{i}/spine")
            seg.append(f"point_cloud_{i}/seg")
            # segments_skel.append(self.load(f"point_cloud_segments_{i}/skel"))
            # segments_skel_gnb.append(self.load(f"point_cloud_segments_{i}/skel_gnb_0"))
            # # segments_skel_gnb.append(self.load(f"point_cloud_segments_{i}/skel_gnb"))
            # for j in range(6):
            #     segments.append(f"point_cloud_segments_{i}/pc_{j}")
            #     segments.append(f"point_cloud_segments_{i}/pc_gnb_{j}")
            #     segments.append(f"point_cloud_segments_{i}/closest_idx_{j}")
            #     segments.append(f"point_cloud_segments_{i}/dist_{j}")

        from chunk_pipeline.utils import object_array

        # skels = object_array(skels)
        # longest_paths = object_array(longest_paths)
        #
        # self.export(
        #     "pc_segments_skel.zip",
        #     [segments_skel, segments_skel_gnb],
        #     ["skel", "skel_gnb"],
        # )
        self.export("pc.zip", idxs + spines + seg, idxs + spines + seg)
        print("done pc")
        self.export("pc_segments.zip", segments, segments)
        self.export("skel.zip", [skels, longest_paths], ["skel", "longest_path"])
        # self.export("pc_segments.zip", segments, segments)
