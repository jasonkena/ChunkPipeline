import numpy as np
from cloudvolume import Vec
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from utils import get_conf
from visualize import read_mappings


def main(conf):
    tq = LocalTaskQueue(parallel=conf.n_jobs_skeletonize)

    mapping = np.load(conf.data.mapping)
    seg_to_trunk, trunk_to_segs = read_mappings(mapping)

    seg_ids = sorted(seg_to_trunk.keys())
    trunk_ids = sorted(trunk_to_segs.keys())
    spine_ids = sorted(list(set(seg_ids) - set(trunk_ids)))

    layer = f"file://{conf.data.output_layer}"

    skeletonize_trunk_tasks = tc.create_skeletonizing_tasks(
        layer, **conf.skeletonize_trunk, object_ids=trunk_ids
    )
    tq.insert(skeletonize_trunk_tasks)
    skeletonize_spines_tasks = tc.create_skeletonizing_tasks(
        layer, **conf.skeletonize_spines, object_ids=spine_ids
    )
    tq.insert(skeletonize_spines_tasks)
    tq.execute()

    merge_tasks = tc.create_unsharded_skeleton_merge_tasks(
        layer, **conf.skeletonize_merge
    )
    tq.insert(merge_tasks)
    tq.execute()


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
