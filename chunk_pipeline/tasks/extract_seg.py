import chunk_pipeline.tasks.chunk as chunk
from chunk_pipeline.utils import extend_bbox


def task_extract_seg(id):
    def inner(cfg, bbox, h5):
        assert "raw" in h5

        bbox = bbox["bbox"].astype(int)
        row = extend_bbox(bbox[id], h5["raw"].shape)

        results = {}
        results["raw"] = chunk.get_seg(h5["raw"], row, filter_id=True)
        if "seg" in h5:
            results["seg"] = (
                chunk.get_seg(h5["seg"], row, filter_id=False) * results["raw"]
            )
        if "spine" in h5:
            results["spine"] = (
                chunk.get_seg(h5["spine"], row, filter_id=False) * results["raw"]
            )

        return results

    return inner
