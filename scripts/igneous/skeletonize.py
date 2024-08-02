from cloudvolume import Vec
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from utils import get_conf


def main(conf):
    tq = LocalTaskQueue(parallel=conf.n_jobs_skeletonize)

    layer = f"file://{conf.data.output_layer}"
    skeletonize_tasks = tc.create_skeletonizing_tasks(layer, **conf.skeletonize)
    tq.insert(skeletonize_tasks)
    tq.execute()

    merge_tasks = tc.create_unsharded_skeleton_merge_tasks(
        layer, **conf.skeletonize_merge
    )
    tq.insert(merge_tasks)
    tq.execute()


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
