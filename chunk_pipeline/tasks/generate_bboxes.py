import chunk_pipeline.tasks.chunk as chunk


def task_bbox(dataset):
    def inner(cfg, h5):
        general = cfg["GENERAL"]
        return {"bbox": chunk.chunk_bbox(h5[dataset], uint_dtype=general["UINT_DTYPE"])}

    return inner
