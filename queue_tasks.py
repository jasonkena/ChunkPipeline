from redis import Redis
from rq import Queue
import numpy as np
from sphere import process_task
import os

if __name__ == "__main__":
    file = "./den_ruilin_v2_16nm.h5"
    bboxes = np.load("bbox.npy")
    q = Queue(
        connection=Redis(
            "redis-11424.c277.us-east-1-3.ec2.cloud.redislabs.com",
            11424,
            password="CR2Oe76Mrl8UTOBop7ulmklF80kXJbc1",
        )
    )
    if not os.path.exists("results"):
        os.makedirs("results")
    for row in bboxes:
        q.enqueue(process_task, job_timeout=-1, args=(file, *row))
